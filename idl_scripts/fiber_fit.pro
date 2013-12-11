
function single_fiber_spectrum,xgrid,ygrid,cube,xcenter,ycenter,width
dist = sqrt((xgrid -xcenter)^2 +(ygrid - ycenter)^2)
fiber_inds = where(dist le width,ct)

for i = 0L,ct-1 do begin
   thisindex = array_indices(dist,fiber_inds[i])
   if i eq 0 then spec = cube[thisindex[0],thisindex[1],*]
   if i ne 0 then spec = spec + cube[thisindex[0],thisindex[1],*]
endfor
return,spec
end

function all_fiber_spectra,xgrid,ygrid,cube,width,offset_vector

n_offsets = (size(offset_vector,/dim))[1]
for i = 0,n_offsets-1 do begin
   this_spec = single_fiber_spectrum(xgrid,ygrid,cube,offset_vector[0,i],offset_vector[1,i],width)
   this_spec = reform(this_spec)
   if i eq 0 then spec_arr = this_spec else spec_arr = [[spec_arr],[this_spec]]
endfor
   
return,spec_arr
end

function compute_fiber_log_likelihood,pars
COMMON FIBER_CONFIG_BLOCK, xgrid_block,ygrid_block,true_spectra_block,fiberwidth_block,offset_vector_block

datacube_model = datacube_simulate(pars,texp=1.,sky=sky_cube,xgrid = xgrid, ygrid = ygrid,$
                                   lambda=lambda,/nonoise)
skysub_datacube = datacube_model - sky_cube
fiber_spectra = all_fiber_spectra(xgrid,ygrid,skysub_datacube,fiberwidth_block,offset_vector_block)
chi = total((skysub_datacube - true_spectra_block)^2/2.,/double)
logL = -chi
return,chi
end

function fiber_mcmc,start_pars,scale,nlinks=nlinks,logL = logL
if n_elements(nlinks) eq 0 then nlinks = 1000.
;start_pars -- the starting parameters vector.
;scale      -- vector of same length as start_pars, which
;              gives the size of the gaussian kernel from which the
;              next parameter in that links is chosen. Set to zero to
;              fix a parameter.


pars = dblarr(n_elements(start_pars), nlinks)
logL = dblarr(nlinks)
pars[*,0] = start_pars
logL[0] = compute_fiber_log_likelihood(start_pars)
for i = 1L,nlinks-1 do begin
   trial_pars = pars + scale * randomn(seed,n_elements(pars))
   trial_logL = compute_fiber_log_likelihood(trial_pars)
   Lratio =  logL[i-1] - trial_logL
   if randomu(seed) lt 1/(1+exp(Lratio)) then begin
      pars[*,i] = trial_pars
      logL[i] = trial_logL
   endif else begin
      pars[*,i] = pars[*,i-1]
      logL[i] = logL[i-1]
   endelse
   if i mod 10 eq 0 then print,'On link '+string(i,form='(I0)')+' of '+string(nlinks,form='(I0)')
endfor
return,pars
end



pro fiber_fit
COMMON FIBER_CONFIG_BLOCK, xgrid_block,ygrid_block,spectra_truth_block,fiberwidth_block,offset_vector_block
;First, make a set of high S/N spectra to use as the template.
;--------------------------------------------------
;Define parameters:
p = fltarr(7)
p[0] = 1.                       ; galaxy Luminosity 
p[1] = 0.5                      ; scale radius
p[2] = !Pi/4.                   ; inclination angle (rad)
p[3] = !Pi/4.                   ; position angle (rad)
p[4] = 0.5                      ; redshift
p[5] = 1.                       ; line intensity
p[6] = 220.                     ; circular velocity
;--------------------------------------------------
datacube_truth = datacube_simulate(p,texp=100.,sky=sky_cube,xgrid = xgrid, ygrid = ygrid,$
                                   lambda=lambda)


;Choose positions for the offset fibers.
offset1 = 4*[-2.2*p[1]/sqrt(2),2.2*p[1]/sqrt(2)]
offset2 = -offset1
offset3 = [0.,0.]
offset_vector = [[offset1],[offset2],[offset3]]
fibersize = 4.

;Sky subtract, and combine the fiber spectra into a single data vector
spec_arr = all_fiber_spectra(xgrid,ygrid,datacube_truth,fibersize,offset_vector)
sky_arr  = all_fiber_spectra(xgrid,ygrid,sky_cube,fibersize,offset_vector)
spec_arr = spec_arr - sky_arr


;configure the fiber position and model fitting
xgrid_block = xgrid & ygrid_block = ygrid
spectra_truth_block = spec_arr
fiberwidth_block = fibersize
offset_vector_block = offset_vector

;Run MCMC. 
start =p
scale = [0., 0., 0.01, 0.01, 0., 0., 0.]
chain = fiber_mcmc(start,scale, logL = logL, nl = 500.)
stop
end

