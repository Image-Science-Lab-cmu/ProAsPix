clear all
close all
addpath('functions')

%%Control knobs
fname = '0723.1436.ButterflyWhite'; %%dataset (check the "data/Processed" folder for other options)
nMeasurements = 4; %%number of image measurements to use. Change change this to between 1 to 16 
measType = 'restored'; %%%Other options: 'simulated', 'measured'
supernum = 500*nMeasurements; %%number of super-pixels. increase linearly with number of measurements
verbose = 2; %0 - no output. 1 - just final results. 2 - intermediate points

if (verbose > 1)
    fprintf('Scene: %s \n', fname);
    fprintf('Number of measurements: %d\n', nMeasurements);
    fprintf('Using measurements type: %s\n', measType);
end


%%Folders
ProcFolder = '../../data/Processed/'; %%SLM aligned processed data folder
RestFolder = '../../data/Restored/'; %%RestoredNet outputs
spectrumFile = '../../resources/Spectrum/0618_Take4.mat'; %%Measurement matrix obtained from spectrometer
load('../../resources/camspecresponse.mat'); %%Camera RGB  spectral response


%%%load measurements
meas = load_processed_data(fname, ProcFolder, RestFolder);

if (verbose > 1)
    figure;
    imshow(meas.guide.^(1/2.2));
    title('RGB guide image');
    drawnow
end

%%%Reconstruct "Full Scan". This serves as ground truth
[hsi_full_scan, hsi_spec, hsi_wvl] = get_full_scan_reconstruction(meas, spectrumFile);


if (verbose > 1)
    figure;
    imagesc(hsi_wvl, 0:255, hsi_spec); colorbar
    title('Measurement operator')
    xlabel('wavelength in [nm]');
    ylabel('SLM index');
    drawnow
end

%%%Set of 16 patterns to use
listIndices = [18 17 19 20 7 8 2 5 81 47 39 33 35 45 43 37 ];
mPatterns = listIndices(1:nMeasurements); %Picking first "nMeasurements" one
if (verbose > 1)
    montage(meas.assort_index(1:128, 1:128, mPatterns), 'BorderSize', [16 16], 'BackgroundColor', 'white');
    title('Measurements patterns being used');
    drawnow
end

%%%Super-pixellation
[L, num] = superpixels(max(0, meas.guide).^(1/2.1), supernum);
if (verbose > 1)
    mask = boundarymask(L);
    img = meas.guide;
    img(find(mask)) = 1;
    figure;
    imshow(img.^(1/2.2));
    title('Super-pixel segmentation');
    drawnow
end

%%%Measurement indices
assort_index = meas.assort_index(:,:,mPatterns);

%%%Pick the appropriate measurement type
switch measType %simulated, measured,restored
    case 'simulated'
        assort_meas = meas.assort_sim(:,:,mPatterns);
    case 'measured'
        assort_meas = meas.assort_meas(:,:,mPatterns);
    case 'restored'
        assort_meas = meas.assort_restored(:,:,mPatterns);   
end
assort_meas = double(assort_meas)/2^16;

%%% Main recosntruction command
tic
hsi_est = reconstruct_rank1_superpixels_v3(assort_meas, assort_index, meas.guide, L, num, hsi_spec, hsi_wvl);
tim = toc;

%%Cleanup
hsi_est(isnan(hsi_est)) = 0;
hsi_est= max(0, hsi_est);
hsi_est = hsi_est/norm(hsi_est(:));

%%Render RGB image from reconstruction
rgbimg = render_rgb_image(hsi_est, hsi_wvl, camspec);
if (verbose > 0)
    subplot 121
    imshow(2*rgbimg/max(rgbimg(:)))
    title('RGB rendered image')
    subplot 122
    imagesc((180/pi)*acos(sum(hsi_est.*hsi_full_scan, 3)./( sqrt(sum(hsi_est.^2, 3)).*sqrt(sum(hsi_full_scan.^2, 3)))));
    colorbar
    axis off
    axis equal
    title('Spectrum angular error in degrees');
    drawnow
end
%%%Get performance metrics against Full Scan
[rsnr, psnr, med_ang] = get_performance_metric(hsi_full_scan, hsi_est);

if (verbose > 0)
    fprintf('Median angle error: %3.3f degrees, PSNR: %3.3f dB\n',  med_ang, psnr);
end