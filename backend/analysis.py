"""Project NightLock: per-band source photometry and diagnostic products.

Run from backend after processing.py --mode align:
  python analysis.py --target barnards_star
  python analysis.py --target barnards_star --band SPHEREx-D1

Each band is analysed independently. Values remain native image units and are
NOT calibrated fluxes. This is a cautious exploratory pipeline, not a validated
SPHEREx science pipeline; check calibration, DQ semantics, PSF and WCS first.
"""
from __future__ import annotations
import argparse, csv, json, logging
from pathlib import Path
from typing import Any
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import processing

ROOT=processing.ROOT
ALIGNED_ROOT=processing.ALIGNED_DIR
OUTPUT_ROOT=processing.PROCESSED_DIR/'analysis'
DEFAULT_APERTURE_PIX=3.0
DETECTION_SIGMA=5.0


def robust_location_scale(data,valid=None):
    a=np.asarray(data,dtype=float); good=np.isfinite(a)
    if valid is not None: good &= np.asarray(valid,dtype=bool)
    v=a[good]
    if v.size<10: raise ValueError('Fewer than 10 valid pixels for background/noise estimate')
    med=float(np.median(v)); sig=1.4826*float(np.median(np.abs(v-med)))
    if not np.isfinite(sig) or sig<=0: sig=float(np.std(v))
    if not np.isfinite(sig) or sig<=0: raise ValueError('Could not estimate positive background noise')
    return med,sig


def read_aligned(path:Path):
    with fits.open(path,memmap=False) as hdul:
        ix={str(h.name).upper():i for i,h in enumerate(hdul)}
        si=ix.get('SCIENCE')
        if si is None: raise ValueError(f'Missing SCIENCE HDU: {path}')
        sci=np.asarray(hdul[si].data,dtype=np.float32).copy(); hdr=hdul[si].header.copy()
        wcs=WCS(hdr,relax=True).celestial
        def get(name):
            i=ix.get(name)
            return np.asarray(hdul[i].data).copy() if i is not None and hdul[i].data is not None else None
        var=get('VARIANCE'); flags=get('FLAGS'); foot=get('FOOTPRINT')
    if not wcs.has_celestial: raise ValueError(f'No celestial WCS: {path}')
    bad=~np.isfinite(sci)
    if foot is not None: bad |= ~np.isfinite(foot)|(foot<=0)
    if var is not None: bad |= ~np.isfinite(var)|(var<=0)
    # Unknown flag bits are preserved but not masked until SPHEREx semantics are confirmed.
    return {'science':sci,'variance':var,'flags':flags,'footprint':foot,'wcs':wcs,'header':hdr,'bad':bad}


def _read_sidecar(path):
    p=path.with_suffix('.meta.json')
    if p.exists():
        try: return json.loads(p.read_text(encoding='utf-8'))
        except Exception as e: logging.warning('Could not parse %s: %s',p,e)
    return {}


def observation_metadata(path,loaded,manifest):
    side=_read_sidecar(path); hdr=loaded['header']; stem=path.stem
    row=manifest.get(str(side.get('plane_id') or hdr.get('PLANEID') or stem),{})
    band=side.get('bandpass') or hdr.get('BANDPASS') or row.get('bandpass') or 'unknown_band'
    mjd=hdr.get('MJD-OBS')
    date=hdr.get('DATE-OBS')
    value=side.get('observation_time') or row.get('observation_time')
    epoch=None
    for v,fmt in ((mjd,'mjd'),(date,None),(value,None)):
        if v in (None,''): continue
        try:
            if fmt: epoch=Time(float(v),format=fmt)
            else:
                try: epoch=Time(float(v),format='mjd') if float(v)>30000 else Time(v)
                except (ValueError,TypeError): epoch=Time(v)
            break
        except Exception: continue
    return {'band':str(band),'plane_id':str(side.get('plane_id') or hdr.get('PLANEID') or stem),'epoch':epoch,'sidecar':side}


def detect_sources(image,bad=None,threshold_sigma=DETECTION_SIGMA,min_separation=3):
    a=np.asarray(image,dtype=float); invalid=~np.isfinite(a)
    if bad is not None: invalid |= np.asarray(bad,dtype=bool)
    bg,noise=robust_location_scale(a,~invalid); work=a.copy(); work[invalid]=-np.inf
    peaks=maximum_filter(work,size=2*min_separation+1,mode='constant',cval=-np.inf)
    yy,xx=np.where((~invalid)&(work==peaks)&(work>bg+threshold_sigma*noise))
    out=[{'x':float(x),'y':float(y),'peak':float(a[y,x]),'snr_peak':float((a[y,x]-bg)/noise)} for x,y in zip(xx,yy)]
    return sorted(out,key=lambda x:x['snr_peak'],reverse=True)


def aperture_photometry(image,x,y,radius=DEFAULT_APERTURE_PIX,bad=None,variance=None,subpixels=5):
    """Fixed-radius subpixel aperture; annular median background and propagated error."""
    a=np.asarray(image,dtype=float); h,w=a.shape
    # Subpixel area weights reduce integer-pixel aperture-size changes as centering shifts.
    yy,xx=np.indices(a.shape); weight=np.zeros(a.shape,dtype=float)
    offsets=(np.arange(subpixels)+0.5)/subpixels-0.5
    for oy in offsets:
        for ox in offsets: weight += (((xx+ox-x)**2+(yy+oy-y)**2)<=radius**2)
    weight/=subpixels**2
    rr=np.hypot(xx-x,yy-y); ann=(rr>=radius+2)&(rr<=radius+5)
    good=np.isfinite(a)
    if bad is not None: good &= ~np.asarray(bad,dtype=bool)
    apw=weight*good; n_eff=float(apw.sum()); n_nom=float(weight.sum())
    annvals=a[ann&good]
    if n_eff<max(3.0,0.7*n_nom) or annvals.size<10:
        return {'flux':None,'flux_error':None,'background':None,'background_sigma':None,'n_aperture_pixels':n_eff,'aperture_valid_fraction':n_eff/max(n_nom,1e-9),'n_background_pixels':int(annvals.size),'quality':'insufficient_valid_aperture_or_annulus'}
    bg=float(np.median(annvals)); _,bgsig=robust_location_scale(annvals)
    # Avoid NaN propagation: IEEE arithmetic makes 0 * NaN equal NaN,
    # so multiplying the full image by zeroed aperture weights is unsafe.
    # Sum only valid pixels that actually contribute to the aperture.
    flux=float(np.sum(apw[good]*(a[good]-bg)))
    # Variance-map term plus uncertainty of median annular background.
    if variance is not None:
        v=np.asarray(variance,dtype=float); vg=(apw>0)&np.isfinite(v)&(v>0)
        var_sum=float(np.sum((apw[vg]**2)*v[vg]))
        # Do not silently claim full uncertainty if variance map is incomplete.
        var_complete=bool(np.all(v[apw>0][np.isfinite(v[apw>0])]>0) and np.count_nonzero(vg)>0)
        err2=var_sum+(n_eff**2)*(np.pi/2)*(bgsig**2)/annvals.size
        quality='variance_plus_background_estimate' if var_complete else 'partial_variance_plus_background_estimate'
    else:
        err2=(bgsig**2)*float(np.sum(apw**2))+(n_eff**2)*(np.pi/2)*(bgsig**2)/annvals.size
        quality='background_noise_estimate_no_variance_map'
    err=float(np.sqrt(max(err2,0)))
    return {'flux':flux,'flux_error':err,'background':bg,'background_sigma':bgsig,'n_aperture_pixels':n_eff,'aperture_valid_fraction':n_eff/max(n_nom,1e-9),'n_background_pixels':int(annvals.size),'quality':quality}


def target_pixel(target,epoch,wcs):
    coord=processing.target_position(target,epoch)
    x,y=wcs.world_to_pixel(coord)
    return float(x),float(y)


def analyse_band(target,band,files,mode,aperture,threshold):
    manifest=processing.read_manifest(); records=[]; images=[]
    for i,path in enumerate(files,1):
        try:
            d=read_aligned(path); md=observation_metadata(path,d,manifest); epoch=md['epoch']
            x,y=target_pixel(target,epoch,d['wcs'])
            phot=aperture_photometry(d['science'],x,y,aperture,d['bad'],d['variance'])
            # Detection catalogue is diagnostic only; target photometry is at propagated catalogue position.
            try: sources=detect_sources(d['science'],d['bad'],threshold)
            except ValueError: sources=[]
            rec={'target':target,'file':path.name,'plane_id':md['plane_id'],'band':band,'observation_time':epoch.isot if epoch else None,'mjd':float(epoch.mjd) if epoch else None,'target_x':x,'target_y':y,'target_flux':phot['flux'],'target_flux_error':phot['flux_error'],'target_background':phot['background'],'background_sigma':phot['background_sigma'],'n_aperture_pixels':phot['n_aperture_pixels'],'aperture_valid_fraction':phot['aperture_valid_fraction'],'n_background_pixels':phot['n_background_pixels'],'photometry_quality':phot['quality'],'source_count':len(sources),'astrometry_note':processing.TARGETS[target]['note']}
            records.append(rec); images.append({'path':path,'data':d,'epoch':epoch,'sources':sources,'record':rec})
        except Exception as e:
            logging.exception('Analysis failed for %s',path); records.append({'target':target,'file':path.name,'band':band,'error':str(e)})
        if i%25==0: logging.info('%s analysed %d/%d',band,i,len(files))
    outdir=OUTPUT_ROOT/target/band; outdir.mkdir(parents=True,exist_ok=True)
    good=[r for r in records if r.get('target_flux') is not None and np.isfinite(r['target_flux'])]
    good.sort(key=lambda r:r['mjd'] if r.get('mjd') is not None else float('inf'))
    flux=np.asarray([r['target_flux'] for r in good],float)
    if flux.size:
        med=float(np.median(flux)); sig=float(1.4826*np.median(np.abs(flux-med)))
        variability={'band':band,'n_epochs':int(len(good)),'median_flux':med,'robust_sigma_flux':sig,'fractional_robust_scatter':float(sig/abs(med)) if med else None,'min_flux':float(np.min(flux)),'max_flux':float(np.max(flux)),'note':'Per-band relative aperture flux in native image units; not absolutely calibrated. Do not interpret scatter as intrinsic variability without calibration/systematics checks.'}
    else: variability={'band':band,'n_epochs':0,'note':'No valid target aperture measurements.'}
    # We intentionally do not fit proper motion from pixel centroids. WCS/association must be validated first.
    motion={'band':band,'n_centroid_epochs':0,'fit_performed':False,'note':'No proper-motion fit: validate WCS distortion, source association, and astrometric reference frame before fitting sky-coordinate motion.'}
    diff_summary=[]
    if mode in ('all','difference') and images:
        ref=images[0]; refim=ref['data']['science']; refw=ref['data']['wcs']
        for item in images:
            # Difference is only meaningful if all images share the same aligned grid.
            a=item['data']['science']; valid=np.isfinite(a)&np.isfinite(refim)&~item['data']['bad']&~ref['data']['bad']
            if a.shape!=refim.shape or np.count_nonzero(valid)<10: continue
            diff=np.full(a.shape,np.nan,dtype=np.float32); diff[valid]=a[valid]-refim[valid]
            name=item['path'].stem+'_difference.fits'
            fits.PrimaryHDU(diff,header=refw.to_header(relax=True)).writeto(outdir/name,overwrite=True)
            diff_summary.append({'file':item['path'].name,'reference_file':ref['path'].name,'difference_file':name,'rms_native_units':float(np.sqrt(np.mean(diff[valid]**2))),'note':'Raw difference; no validated PSF matching or photometric scaling applied.'})
    with (outdir/'photometry.csv').open('w',newline='',encoding='utf-8') as f:
        fields=sorted({k for r in records for k in r}); wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); wr.writerows(records)
    (outdir/'photometry.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (outdir/'variability.json').write_text(json.dumps(variability,indent=2),encoding='utf-8')
    (outdir/'motion_summary.json').write_text(json.dumps(motion,indent=2),encoding='utf-8')
    (outdir/'source_catalogue.json').write_text(json.dumps([{'file':im['path'].name,'sources':im['sources']} for im in images],indent=2),encoding='utf-8')
    (outdir/'difference_summary.json').write_text(json.dumps(diff_summary,indent=2),encoding='utf-8')
    if good:
        fig,ax=plt.subplots(figsize=(10,5)); times=[r['mjd'] for r in good]; vals=[r['target_flux'] for r in good]; errs=[r['target_flux_error'] for r in good]
        if all(t is not None for t in times): ax.errorbar(times,vals,yerr=errs,fmt='o',markersize=3,capsize=2); ax.set_xlabel('Modified Julian Date')
        else: ax.errorbar(np.arange(len(vals)),vals,yerr=errs,fmt='o',markersize=3,capsize=2); ax.set_xlabel('Observation index (MJD unavailable)')
        ax.set_ylabel('Background-subtracted aperture flux (native units)'); ax.set_title(f'{target} — {band}'); ax.grid(True,alpha=.25); fig.tight_layout(); fig.savefig(outdir/'light_curve.png',dpi=160); plt.close(fig)
    summary={'target':target,'band':band,'aligned_images_found':len(files),'images_analysed':len(images),'images_with_errors':sum('error' in r for r in records),'variability':variability,'motion':motion,'output_directory':str(outdir),'scientific_caveats':['Per-band native-unit aperture photometry; no absolute calibration.','Background uncertainty is estimated from the local annulus; variance-map uncertainties depend on input map semantics and reprojection.','Unknown DQ flag bits are retained but not masked.','No PSF matching or photometric scaling is applied to difference images.','No proper motion is fitted until WCS and source association are independently validated.']}
    (outdir/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    logging.info('Band %s products written to %s',band,outdir); return summary


def main():
    ap=argparse.ArgumentParser(description='NightLock per-band photometry analysis')
    ap.add_argument('--target',required=True,choices=sorted(processing.TARGETS)); ap.add_argument('--band',default=None,help='Optional aligned band folder name')
    ap.add_argument('--mode',choices=['all','detect','difference'],default='all'); ap.add_argument('--aperture',type=float,default=DEFAULT_APERTURE_PIX); ap.add_argument('--threshold',type=float,default=DETECTION_SIGMA); ap.add_argument('--log-level',default='INFO')
    a=ap.parse_args(); logging.basicConfig(level=getattr(logging,a.log_level.upper(),logging.INFO),format='%(levelname)s: %(message)s')
    root=ALIGNED_ROOT/a.target
    if a.band:
        dirs=[root/a.band]
    else:
        dirs=sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    dirs=[d for d in dirs if any(p.is_file() and p.suffix.lower()=='.fits' for p in d.glob('*.fits'))]
    if not dirs: raise SystemExit(f'No aligned band directories found under {root}. Run processing.py --target {a.target} --mode align first.')
    summaries=[]
    for d in dirs:
        fs=sorted(p for p in d.glob('*.fits') if p.is_file())
        summaries.append(analyse_band(a.target,d.name,fs,a.mode,a.aperture,a.threshold))
    logging.info('Completed %d band(s), %d image analyses.',len(summaries),sum(x['images_analysed'] for x in summaries))

if __name__=='__main__': main()
