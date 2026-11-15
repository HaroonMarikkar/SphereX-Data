"""Project NightLock: FITS metadata preservation, coverage and alignment.

Run from backend:
  python processing.py --target barnards_star --mode coverage
  python processing.py --target barnards_star --mode align
  python processing.py --target barnards_star --mode all
  python processing.py --target barnards_star --band SPHEREx-D1 --mode all

Important: target coordinates and proper motions below remain provisional. Variance
reprojection is approximate; SPHEREx DQ bit meanings are deliberately not guessed.
"""
from __future__ import annotations
import argparse, csv, json, logging, re
from pathlib import Path
from typing import Any
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.time import Time
from astropy.wcs import WCS
from reproject import reproject_interp

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data_pipeline' / 'downloads' / 'challenge_targets'
MANIFEST_PATH = DATA_DIR / 'observation_manifest.csv'
PROCESSED_DIR = ROOT / 'data_pipeline' / 'processed'
ALIGNED_DIR = PROCESSED_DIR / 'aligned'
COVERAGE_DIR = PROCESSED_DIR / 'challenge_dataset'
TARGETS = {
 'barnards_star': dict(ra=269.452077, dec=4.693365, pm_ra_cosdec=-801.551, pm_dec=10362.394, note='Provisional working astrometry; verify catalogue and reference epoch.'),
 'epsilon_indi_b': dict(ra=331.044146, dec=-56.782844, pm_ra_cosdec=3966.661, pm_dec=-2536.192, note='Motion approximates primary system; verify component astrometry.'),
 'wise_0855': dict(ra=133.795129, dec=-7.245147, pm_ra_cosdec=0.0, pm_dec=0.0, note='Proper motion unverified and set to zero; do not use for precision positioning.'),
 'rw_aurigae': dict(ra=76.956526, dec=30.401438, pm_ra_cosdec=7.23, pm_dec=-20.33, note='Provisional working astrometry; verify catalogue and reference epoch.'),
 'r_draconis': dict(ra=248.167611, dec=66.754979, pm_ra_cosdec=-20.643, pm_dec=6.149, note='Provisional working astrometry; verify catalogue and reference epoch.'),
}
REFERENCE_EPOCH = Time('J2000.0')


def read_manifest() -> dict[str, dict[str, str]]:
    rows = {}
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                pid = str(row.get('plane_id') or '').strip()
                if pid: rows[Path(pid).stem] = row
    else: logging.warning('Manifest not found: %s', MANIFEST_PATH)
    return rows


def identify_hdus(hdul):
    named = {'SCIENCE': None, 'SCI': None, 'IMAGE': None, 'VARIANCE': None, 'VAR': None, 'FLAGS': None, 'DQ': None, 'MASK': None}
    for i,h in enumerate(hdul):
        if h.data is not None and np.ndim(h.data)==2:
            n=str(h.name).upper()
            if n in named: named[n]=i
    sci=next((named[n] for n in ('SCIENCE','SCI','IMAGE') if named[n] is not None),None)
    if sci is None:
        cand=[(np.asarray(h.data).size,i) for i,h in enumerate(hdul) if h.data is not None and np.ndim(h.data)==2]
        if not cand: raise ValueError('No 2D image HDU found')
        sci=max(cand)[1]
    var=next((named[n] for n in ('VARIANCE','VAR') if named[n] is not None),None)
    dq=next((named[n] for n in ('FLAGS','DQ','MASK') if named[n] is not None),None)
    return sci,var,dq


def _header_with_primary(primary, science):
    # Preserve all science-header cards; fill absent WCS/instrument cards from primary.
    h=science.copy()
    for card in primary.cards:
        if card.keyword not in ('', 'COMMENT', 'HISTORY') and card.keyword not in h:
            try: h.append(card)
            except Exception: pass
    return h


def _time(meta, manifest_time=None):
    for value, fmt in ((meta.get('mjd_obs'),'mjd'), (meta.get('date_obs'),None), (manifest_time,None)):
        if value in (None,''): continue
        try:
            if fmt: return Time(float(value),format=fmt)
            try:
                v=float(value)
                if v>30000: return Time(v,format='mjd')
            except (TypeError,ValueError): pass
            return Time(value)
        except Exception: continue
    return None


def load_fits(path: str|Path, manifest_row: dict|None=None) -> dict[str,Any]:
    path=Path(path)
    if not path.is_file() or path.name.lower().endswith(('.part','.fits.part')): raise ValueError(f'Not a complete FITS file: {path}')
    with fits.open(path,memmap=False,lazy_load_hdus=False) as hdul:
        hdul.verify('exception'); si,vi,fi=identify_hdus(hdul)
        sh=hdul[si]; header=_header_with_primary(hdul[0].header,sh.header)
        # relax=True retains supported non-standard/SIP WCS cards rather than silently dropping them.
        wcs=WCS(header,relax=True).celestial
        if not wcs.has_celestial: raise ValueError(f'No celestial WCS: {path}')
        sci=np.asarray(sh.data,dtype=np.float32).copy()
        var=np.asarray(hdul[vi].data,dtype=np.float32).copy() if vi is not None else None
        flags=np.asarray(hdul[fi].data).copy() if fi is not None else None
        if sci.ndim!=2 or (var is not None and var.shape!=sci.shape) or (flags is not None and flags.shape!=sci.shape): raise ValueError(f'Image/variance/flag shape mismatch: {path}')
        mrow=manifest_row or {}
        meta={
          'filename':path.name,'plane_id':str(mrow.get('plane_id') or path.stem),
          'bandpass':str(mrow.get('bandpass') or header.get('BANDPASS') or header.get('FILTER') or path.parent.name),
          'observation_time':mrow.get('observation_time'),'mjd_obs':header.get('MJD-OBS'),
          'date_obs':header.get('DATE-OBS'),'exptime':header.get('EXPTIME'),
          'bunit':header.get('BUNIT'),'instrument':header.get('INSTRUME'),
          'telescope':header.get('TELESCOP'),'object':header.get('OBJECT'),
          'science_hdu':si,'variance_hdu':vi,'flags_hdu':fi,'shape':list(sci.shape),
        }
    return {'science':sci,'variance':var,'flags':flags,'wcs':wcs,'header':header,'metadata':meta}


def create_quality_mask(science,variance=None,flags=None,excluded_flag_bits=None):
    bad=~np.isfinite(science)
    if variance is not None: bad |= ~np.isfinite(variance) | (variance<=0)
    # Only mask flag bits when their meanings have been verified for this data release.
    if flags is not None and excluded_flag_bits is not None: bad |= (flags.astype(np.int64)&int(excluded_flag_bits))!=0
    return bad


def target_position(target,epoch=None):
    d=TARGETS[target]
    c=SkyCoord(ra=d['ra']*u.deg,dec=d['dec']*u.deg,pm_ra_cosdec=d['pm_ra_cosdec']*u.mas/u.yr,pm_dec=d['pm_dec']*u.mas/u.yr,obstime=REFERENCE_EPOCH,frame='icrs')
    if epoch is None: return c
    try: return c.apply_space_motion(new_obstime=epoch)
    except Exception as e: logging.warning('Could not propagate %s to epoch (%s); using catalogue position.',target,e); return c


def safe_band_name(value):
    s=re.sub(r'[^A-Za-z0-9_.-]+','_',str(value or 'unknown_band')).strip('._')
    return s or 'unknown_band'


def find_target_files(target,band=None):
    d=DATA_DIR/target
    if not d.exists(): raise FileNotFoundError(d)
    files=sorted(p for p in d.rglob('*.fits') if p.is_file())
    if band: files=[p for p in files if band.lower() in str(p).lower()]
    return files


def _row_for(path,manifest): return manifest.get(path.stem,{})


def check_coverage(path,target,manifest):
    row=_row_for(path,manifest); loaded=load_fits(path,row); epoch=_time(loaded['metadata'],row.get('observation_time'))
    coord=target_position(target,epoch); x,y=loaded['wcs'].world_to_pixel(coord); h,w=loaded['science'].shape
    inside=bool(np.isfinite(x) and np.isfinite(y) and 0<=x<w and 0<=y<h)
    return {'target':target,'file':str(path),'plane_id':loaded['metadata']['plane_id'],'bandpass':loaded['metadata']['bandpass'],'observation_time':epoch.isot if epoch else None,'mjd':float(epoch.mjd) if epoch else None,'ra_deg':float(coord.ra.deg),'dec_deg':float(coord.dec.deg),'x_pixel':float(x),'y_pixel':float(y),'inside':inside,'astrometry_note':TARGETS[target]['note']}


def run_coverage(target,band,files,manifest):
    rows=[]
    for i,p in enumerate(files,1):
        try: rows.append(check_coverage(p,target,manifest))
        except Exception as e: logging.exception('Coverage failed for %s',p); rows.append({'target':target,'file':str(p),'error':str(e),'inside':False})
        if i%25==0: logging.info('Coverage checked %d/%d files',i,len(files))
    COVERAGE_DIR.mkdir(parents=True,exist_ok=True); suffix='_'+safe_band_name(band) if band else ''
    for ext,writer in [('json',lambda p:p.write_text(json.dumps(rows,indent=2),encoding='utf-8')),('csv',None)]:
        p=COVERAGE_DIR/f'{target}{suffix}_coverage.{ext}'
        if writer: writer(p)
        else:
            with p.open('w',newline='',encoding='utf-8') as f:
                fields=sorted({k for r in rows for k in r}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def _band_groups(files,manifest):
    groups={}
    for p in files:
        row=_row_for(p,manifest)
        # Manifest bandpass is authoritative; folder name is fallback only.
        band=row.get('bandpass') or p.parent.name
        groups.setdefault(safe_band_name(band),[]).append(p)
    return groups


def build_reference_grid(reference_path,target,size_arcmin=6.0,manifest_row=None):
    d=load_fits(reference_path,manifest_row); epoch=_time(d['metadata'],(manifest_row or {}).get('observation_time'))
    pos=target_position(target,epoch)
    cut=Cutout2D(d['science'],pos,size_arcmin*u.arcmin,wcs=d['wcs'],mode='partial',fill_value=np.nan)
    return cut.wcs,cut.data.shape


def align_image(input_path,reference_wcs,output_shape,output_path,manifest_row=None):
    d=load_fits(input_path,manifest_row); meta=d['metadata']
    sci,foot=reproject_interp((d['science'],d['wcs']),reference_wcs,shape_out=output_shape,order='bilinear')
    sci=np.asarray(sci,dtype=np.float32); foot=np.asarray(foot,dtype=np.float32)
    var=None; flags=None
    if d['variance'] is not None:
        var,vfoot=reproject_interp((d['variance'],d['wcs']),reference_wcs,shape_out=output_shape,order='nearest-neighbor')
        var=np.asarray(var,dtype=np.float32); var[(vfoot<=0)|~np.isfinite(var)]=np.nan
    if d['flags'] is not None:
        f,ffoot=reproject_interp((d['flags'].astype(np.float32),d['wcs']),reference_wcs,shape_out=output_shape,order='nearest-neighbor')
        flags=np.where(ffoot>0,np.nan_to_num(f),0).astype(np.int32)
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    sh=fits.ImageHDU(sci,name='SCIENCE'); sh.header.update(reference_wcs.to_header(relax=True))
    sh.header['SRCFILE']=input_path.name; sh.header['BANDPASS']=str(meta['bandpass'])[:68]
    sh.header['PLANEID']=str(meta['plane_id'])[:68]
    if meta.get('mjd_obs') not in (None,''):
        try: sh.header['MJD-OBS']=float(meta['mjd_obs'])
        except (TypeError,ValueError): pass
    if meta.get('date_obs'): sh.header['DATE-OBS']=str(meta['date_obs'])[:68]
    if meta.get('exptime') is not None:
        try: sh.header['EXPTIME']=float(meta['exptime'])
        except (TypeError,ValueError): pass
    sh.header['VAREPROJ']='NEAREST' if var is not None else 'NONE'
    hdus=[fits.PrimaryHDU(),sh]
    if var is not None: hdus.append(fits.ImageHDU(var,name='VARIANCE'))
    if flags is not None: hdus.append(fits.ImageHDU(flags,name='FLAGS'))
    hdus.append(fits.ImageHDU(foot,name='FOOTPRINT'))
    fits.HDUList(hdus).writeto(out,overwrite=True,checksum=True)
    sidecar={**meta,'source_path':str(input_path),'aligned_path':str(out),'variance_reprojection':'nearest-neighbour approximation' if var is not None else None,'flag_semantics_applied':False}
    out.with_suffix('.meta.json').write_text(json.dumps(sidecar,indent=2,default=str),encoding='utf-8')
    return {'input':str(input_path),'output':str(out),'bandpass':meta['bandpass'],'plane_id':meta['plane_id'],'mjd_obs':meta.get('mjd_obs'),'valid_fraction':float(np.mean((foot>0)&np.isfinite(sci))),'shape':list(output_shape)}


def run_alignment(target,band,files,size_arcmin,manifest):
    reports=[]
    groups={safe_band_name(band):files} if band else _band_groups(files,manifest)
    for band_name,group in groups.items():
        if not group: continue
        refrow=_row_for(group[0],manifest)
        refwcs,shape=build_reference_grid(group[0],target,size_arcmin,refrow)
        outdir=ALIGNED_DIR/target/band_name; outdir.mkdir(parents=True,exist_ok=True)
        local=[]
        for i,p in enumerate(group,1):
            try: local.append(align_image(p,refwcs,shape,outdir/p.name,_row_for(p,manifest)))
            except Exception as e: logging.exception('Alignment failed for %s',p); local.append({'input':str(p),'error':str(e),'bandpass':band_name})
            if i%25==0: logging.info('%s aligned %d/%d',band_name,i,len(group))
        (outdir/'alignment_report.json').write_text(json.dumps(local,indent=2),encoding='utf-8'); reports.extend(local)
        logging.info('Band %s: aligned %d/%d',band_name,sum('output' in x for x in local),len(local))
    return reports


def main():
    ap=argparse.ArgumentParser(description='NightLock FITS processing and alignment')
    ap.add_argument('--target',required=True,choices=sorted(TARGETS)); ap.add_argument('--band',default=None,help='Optional bandpass/folder filter, e.g. SPHEREx-D1')
    ap.add_argument('--mode',choices=['coverage','align','all'],default='all'); ap.add_argument('--size',type=float,default=6.0); ap.add_argument('--log-level',default='INFO')
    a=ap.parse_args(); logging.basicConfig(level=getattr(logging,a.log_level.upper(),logging.INFO),format='%(levelname)s: %(message)s')
    files=find_target_files(a.target,a.band)
    if not files: raise SystemExit('No matching local .fits files found.')
    logging.info('Found %d FITS files for %s',len(files),a.target); manifest=read_manifest()
    if a.mode in ('coverage','all'):
        rows=run_coverage(a.target,a.band,files,manifest); logging.info('Coverage: %d/%d inside',sum(bool(r.get('inside')) for r in rows),len(rows))
    if a.mode in ('align','all'):
        rep=run_alignment(a.target,a.band,files,a.size,manifest); logging.info('Alignment: %d/%d files',sum('output' in r for r in rep),len(rep))
if __name__=='__main__': main()
