function spider,x,y,inclination_angle
  r = sqrt(x^2+y^2)
  theta = atan(y,x)
  velocity = sin(inclination_angle)*sin(theta)
  return,velocity
end


function disk_velocity_field,x,y,vmax,scale,inclination_angle,position_angle,xcen=xcen,ycen=ycen
  if ~keyword_set(xcen) then xcen = 0.
  if ~keyword_set(ycen) then ycen = 0.

  x2 = x*cos(position_angle) + y*sin(position_angle)
  y2 =-x*sin(position_angle) + y*cos(position_angle)
  r = sqrt(x2^2+y2^2)
  vscale = spider(x2,y2,inclination_angle)
  
  vlos = vmax*vscale*atan(r/scale)
  
  return,vlos
end

function exp_2d,x,y,r0,inclination,xcen=xcen,ycen=ycen,nu=nu,thick=thick
  if n_elements(nu) eq 0 then nu = 0.5
  r = sqrt(x^2+y^2)/r0
  phi_r = atan(y,x)
  eps = sqrt(1d - (1d - thick^2)*cos(inclination)^2)
  u = r/r0 *sqrt((1+eps*cos(2*phi_r)))/sqrt(1-eps^2)
  fnu = (u/2.)^nu*beselk(u,nu)/gamma(nu+1.)
  return,fnu/r0^2
end

function disk_galaxy_image,x,y,r0,inclination,position_angle,rcut=rcut,xcen=xcen,ycen=ycen,thick=thick,nu=nu
  if n_elements(thick) eq 0 then thick=0.19
  if ~keyword_set(xcen) then xcen=0.
  if ~keyword_set(ycen) then ycen=0.
  x = x-xcen & y = y-ycen
  x2 = x*cos(position_angle) + y*sin(position_angle)
  y2 =-x*sin(position_angle) + y*cos(position_angle)
  image = exp_2d(x2,y2,r0,inclination,xcen=xcen,ycen=ycen,nu=nu,thick=thick)
  return,image
end

function atmospheric_transmission,lambda_in, reset=reset
COMMON atm_BLOCK,lambda_trans,atm_trans
if (n_elements(atm_trans) eq 0) OR keyword_set(reset) then begin
   file = './Atm_Transmission/atmtrans_default.dat'
   readcol,file,l,atmt
   lambda_trans = l
   atm_trans = atmt
endif
trans = interpol(atm_trans,lambda_trans,lambda_in,/lsq)
return,trans
end

function galaxy_spectrum,lambda_in,z,atm_trans = atm_transmission, reset=reset
common galaxy_template_block,lambda_template,template_spectrum
;dist should be in Mpc
if (n_elements(template_spectrum) eq 0) OR keyword_set(reset) then begin
   template_struct = mrdfits('Galaxy_Spectra/kcorrect-templates.fits',1,/silent)
   lambda_template = template_struct.lambda
   template_spectrum = template_struct.spec[*,1]*1e-8
endif
z_fid = .50 ; -- which corresponds to this cosmological redshift; we need this for surface-brightness dimming.
d_fid = lumdist(z_fid,H0=67.3, omega_m = 0.315, Lambda0 = (1-0.315),/silent)
dist = lumdist(z,H0=67.3, omega_m = 0.315, Lambda0 = (1-0.315),/silent)
spec = interpol(template_spectrum,lambda_template,lambda_in) * (d_fid / dist)^2 * (1+z_fid)^4 /(1+z)^4
if n_elements(atm_transmission) eq 0 then begin
   atm_transmission = atmospheric_transmission(lambda_in) 
endif
spec = spec * atm_transmission
return,spec * (d_fid / dist)^2 * (1+z_fid)^4 /(1+z)^4
end


function sky_spectrum,lambda_in,reset=reset
common galaxy_sky_block,lambda_sky,sky_spectrum
if (n_elements(sky_spectrum) eq 0) OR keyword_set(reset) then begin
   readcol,'Sky_Spectra/kpno_sky.txt',l,sky
   lambda_sky = l
   sky_spectrum = 10.^((21.572-sky)/2.5)*1e-17
endif
sky_interp = interpol(sky_spectrum,lambda_sky,lambda_in,/lsq) ;This is in erg/s/cm^2/arcsec^2
return,sky_interp
end


function seeing_kernel_fft,ngrid,b=b
if n_elements(b) eq 0 then b=0.025 ;This gets us a ~3 pixel seeing disk


;We _always_ have an even number of grid elements.
kx = shift((findgen(ngrid) - (ngrid/2.-1)),-(ngrid/2-1.)) # replicate(1.,ngrid)
ky = transpose(kx)
k = sqrt(kx^2+ky^2)
Tk = exp(-abs(k*b)^(5./3.))
Tk = Tk/total(Tk,/double)
seeing = real_part(shift(fft(Tk,/inv),ngrid/2.-1,ngrid/2.-1))
seeing = seeing / total(seeing)
return,seeing
end


function apply_seeing,cube
;Blurs the 2D image of the galaxy by some finite seeing model.
cube_convol = cube * 0d
nlambda = (size(cube,/dim))[2]
ngrid = (size(cube,/dim))[1]
kernel = seeing_kernel_fft(ngrid)
for i=0L,nlambda-1 do begin
   cube_convol[*,*,i] = convol_fft(cube[*,*,i],kernel,kernel_fft = kernel_fft)
endfor
return,cube_convol
end

function instrumental_resolution,cube, lambda, R

npix = (size(cube,/dim))[0]
cube_resolved = cube * 0.
pixscale = lambda[1] - lambda[0]
dlambda =  median(lambda) / R
smoothing_length = dlambda / pixscale

for ii = 0L,npix-1 do begin
   for jj = 0L,npix-1 do begin
      spec = reform(cube[ii,jj,*])
      cube_resolved[ii,jj,*] = gauss_smooth(smooth(spec,smoothing_length))
   endfor
endfor

return, cube_resolved
end


function slit,xgrid,ygrid,cube,lambda_obs,slit_angle,slit_struct=slit_struct
  if n_elements(slit_struct) eq 0 then begin
     slit_struct = create_struct('lambda_min',5000.,'lambda_max',6000.,'delta_lambda',1.,$
                                 'delta_x',0.1,'slit_width',2.5)     
  endif
  nlambda_cube = n_elements(lambda_obs)
  xgrid_rot = xgrid * cos(slit_angle) + ygrid*sin(slit_angle)
  ygrid_rot = ygrid * cos(slit_angle) - xgrid * sin(slit_angle)

  ;Set all the flux outside of the slit to zero.
  cube_slit = cube*0.
  for i = 0,nlambda_cube-1 do cube_slit[*,*,i] = cube[*,*,i] * (abs(xgrid_rot) lt slit_struct.slit_width)
  slitspec = total(cube_slit,2)
  return,slitspec
end


function add_noise_poisson,cube,lambda,texp,diameter=diameter
;Assume that cube is in erg/s/cm^2, texp in s, and lambda in Angstroms
;Then, the number photon counts is cube[lambda] / (h * c / lambda)
;But this must be scaled up by the collecting area.
;assume 'diameter' is in meters
if n_elements(diameter) eq 0 then diameter  = 8.
area = !pi * (diameter*100.)^2

h = 6.62607e-27 ;erg s
c = 3e10 ;cm/s
nlambda = n_elements(lambda)
size = size(cube[*,*,0],/dim)
new_cube = cube
for i = 0L,nlambda-1 do begin
   counts_2d = cube[*,*,i] / (h*c/lambda[i]) *  area
   for ii = 0L,size[0]-1 do begin
      for jj=0L,size[1]-1 do begin
         if counts_2d[ii,jj] gt 0 then $
            counts_2d[ii,jj] = randomn(seed,poisson=counts_2d[ii,jj],/double)
      endfor
   endfor
   new_cube[*,*,i] = counts_2d * (h*c/lambda[i]) / area
endfor
return,new_cube
end

function add_noise_gaussian,cube,lambda,texp,diameter=diameter,nonoise=nonoise
;Assume that cube is in erg/s/cm^2, texp in s, and lambda in Angstroms
;Then, the number photon counts is cube[lambda] / (h * c / lambda)
;But this must be scaled up by the collecting area.
;assume 'diameter' is in meters
if n_elements(diameter) eq 0 then diameter  = 8.
area = !pi * (diameter*100.)^2

h = 6.62607e-27 ;erg s
c = 3e10 ;cm/s
lambda_cm = lambda * 1e-8
nlambda = n_elements(lambda)
npix = ( size(cube[*,*,0],/dim))[0]
new_cube = cube
for i = 0L,nlambda-1 do begin
   counts_2d = cube[*,*,i] / (h*c/lambda_cm[i]) *  area * texp
   if ~keyword_set(nonoise) then begin
      new_cube[*,*,i] = (sqrt(counts_2d)*randomn(seed,npix,npix) + counts_2d) * (h*c/lambda_cm[i]) / area / texp
   endif else begin
      new_cube[*,*,i] = counts_2d * (h*c/lambda_cm[i]) / area / texp
   endelse
endfor
return,new_cube
end




function datacube_simulate,p,texp=texp,sky_cube = sky_noisy, image = image, lambda = lambda_obs, $
                           xgrid = xgrid, ygrid = ygrid, nonoise=nonoise
time = systime(1)
if n_elements(texp) eq 0 then texp = 1000.; exposure time
;Define a parameters vector:
;--------------------------------------------------
;These first parameters are for the image.
;--------------------------------------------------
;p[0] = galaxy Luminosity
;p[1] = scale radius
;p[2] = inclination angle (rad)
;p[3] = position angle (rad)
;--------------------------------------------------
;These next parameters create the spectrum.
;--------------------------------------------------
;p[4] = redshift
;p[5] = line intensity.
;p[6] = circular velocity
;--------------------------------------------------
if n_elements(p) eq 0 then begin
   p = fltarr(7)
   p[0] = 1.   ; galaxy Luminosity 
   p[1] = 0.5    ; scale radius
   p[2] = !Pi/4. ; inclination angle (rad)
   p[3] = !Pi/4. ; position angle (rad)
   p[4] = 0.5    ; redshift
   p[5] = 1.  ; line intensity
   p[6] = 200.   ; circular velocity
endif


;Build the surface-brightness map. Units should be in arcsec
pixscale = 0.25 ;arcsec per pixel
Resolution = 2000. ; spectral resolution: R = lambda/dlambda
npix = 64.
xgrid = findgen(npix) # replicate(1.,npix)
xgrid = xgrid - mean(xgrid)
ygrid = transpose(xgrid)

;Build the image grid that will hold the galaxy.
;Note that vlos is in km/s. Positive sign means away from the observer
image = disk_galaxy_image(xgrid,ygrid,p[1]/pixscale,p[2],p[3])*p[0]
vlos = disk_velocity_field(xgrid,ygrid,p[6],p[1]/pixscale,p[2],p[3])
c = 300000. ; Speed of light, km/s

;Next, construct the data cube. First, set the baseline wavelength
;range:
nlambda_obs = 1000.
lambda_obs_min = 6000.
lambda_obs_max = 7000.
lambda_obs = lambda_obs_min + findgen(nlambda_obs)/float(nlambda_obs-1.)*(lambda_obs_max - lambda_obs_min)
lambda_rest = lambda_obs / (1. +p[4]) ;Shift into the rest frame of the galaxy

;Make the data cube.
tcube = systime(1)
cube = dblarr(npix,npix,nlambda_obs)
for i = 0L,npix-1 do begin
   for j = 0L,npix-1 do begin
      reset = (i eq 0) AND (j eq 0)
      zscale = vlos[i,j]/c
      cube[i,j,*] = galaxy_spectrum(lambda_rest*(1+zscale),p[4],atm=atm, reset = reset) * p[5] *image[i,j]/total(image)
   endfor
endfor


;Blur by the seeing.
tblur = systime(1)
cube_smeared = apply_seeing(cube)


;Add the sky flux.
tsky = systime(1)
skyspec =  sky_spectrum(lambda_obs)
sky_cube = cube * 0.
for i = 0,npix-1 do begin
   for j=0,npix-1 do begin
      cube_smeared[i,j,*] = cube_smeared[i,j,*]  + skyspec
      sky_cube[i,j,*] = skyspec
   endfor
endfor

;Blur to instrumental resolution.
tres = systime(1)
cube_resolved = instrumental_resolution(cube_smeared,lambda_obs, Resolution)
sky_resolved = instrumental_resolution(sky_cube,lambda_obs, Resolution)

;Add noise.
cube_noisy = add_noise_gaussian(cube_resolved,lambda_obs,texp, nonoise=nonoise)
sky_noisy = add_noise_gaussian(sky_resolved,lambda_obs,texp, nonoise=nonoise)

return,cube_noisy
end

;Next, measure a fiber spectrum.
tfiber = systime(1)
offset1 = 4*[-2.2*p[1]/sqrt(2),2.2*p[1]/sqrt(2)]
offset2 = 4*[-2.2*p[1]/sqrt(2),-2.2*p[1]/sqrt(2)]
fibersize = 4.
spec1 = fiber(xgrid,ygrid,cube_noisy,offset1[0],offset1[1],fibersize)
spec2 = fiber(xgrid,ygrid,cube_noisy,offset2[0],offset2[1],fibersize)
speccen = fiber(xgrid,ygrid,cube_noisy,0.,0.,fibersize)
sky_noisy_fiber = fiber(xgrid,ygrid,sky_noisy,0.,0.,fibersize)

;Finally, display the disk galaxy image, and show the fiber and slit
;footprints and spectra.

psopen,'Plots/TF_fiber_slit_demo',xsize=6,ysize=6,/inches,/color
prepare_plots,/color
loadct,0,/silent
display,image,xgrid[*,0],ygrid[0,*],/aspect,max=max(image)*1.05,min=0.
loadct,6,/silent
dist1 = sqrt((xgrid-offset1[0])^2 + (ygrid - offset1[1])^2)
dist2 = sqrt((xgrid-offset2[0])^2 + (ygrid - offset2[1])^2)
distcen = sqrt((xgrid)^2 + (ygrid)^2)

contour,dist1,xgrid,ygrid,/overplot,level=4.,color=20
contour,dist2,xgrid,ygrid,/overplot,level=4.,color=100
contour,distcen,xgrid,ygrid,/overplot,level=4.,color=150

plot,lambda_obs,speccen-sky_noisy_fiber,xtitle='wavelength ('+string(197B)+')',ytitle='flux',charsize=1.25,thick=2.,/nodata
oplot,lambda_obs,speccen-sky_noisy_fiber,color=150
oplot,lambda_obs,spec2-sky_noisy_fiber,color=100
oplot,lambda_obs,spec1-sky_noisy_fiber,color=20


tslit = systime(1)
slitspec_p = slit(xgrid,ygrid,cube_noisy,lambda_obs,p[3],slit = slit_struct)
slitspec_c = slit(xgrid,ygrid,cube_noisy,lambda_obs,p[3]+!Pi/2.,slit = slit_struct)


loadct,0,/silent
display,image,/aspect,max=max(image)*1.05,min=0.
loadct,6,/silent
xgrid_rot_p = xgrid * cos(p[3]) + ygrid*sin(p[3])
xgrid_rot_c = xgrid * cos(p[3]+!Pi/2.) + ygrid*sin(p[3]+!Pi/2.)
contour,abs(xgrid_rot_p),/overplot,level=slit_struct.slit_width,color=200
contour,abs(xgrid_rot_c),/overplot,level=slit_struct.slit_width,color=200

;Do a poor man's sky subtraction.
sky_region_p = (slitspec_p[5,*] + slitspec_p[-5,*])/2.
sky_region_c = (slitspec_c[5,*] + slitspec_c[-5,*])/2.
sky_subtracted_p = slitspec_p*0.
sky_subtracted_c = slitspec_c*0.
for i =0,npix-1 do begin
   sky_subtracted_p[i,*] = slitspec_p[i,*]  - sky_region_p
   sky_subtracted_c[i,*] = slitspec_c[i,*]  - sky_region_c
endfor
loadct,0,/silent
display,sky_subtracted_p,findgen(npix),lambda_obs,/silent,$
        ytitle='wavelength ('+string(197B)+')',charsize=1.5,min=0.

display,sky_subtracted_c,findgen(npix),lambda_obs,/silent,$
        ytitle='wavelength ('+string(197B)+')',charsize=1.5,min=0.

psclose
prepare_plots,/reset

stop
end
