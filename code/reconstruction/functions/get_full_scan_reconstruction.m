function [hsi_full_scan, hsi_spec, hsi_wvl] = get_full_scan_reconstruction(meas, spectrum_file, tt);

if ~exist('tt')
      %spectral calibration
    tt = '1/linear';
end

switch tt
    case '1/linear'
        wmin= 420; wmax = 950; wnum = floor((wmax-wmin)/10);
        wscaling = '1/linear'; %'linear'; %1/linear
    case 'linear'
        wmin= 420; wmax = 950; wnum = floor((wmax-wmin)/10);
        wscaling = 'linear'; %'linear'; %1/linear
    case 'special'
        wmin= 420; wmax = 950;
        wscaling = 'special'; wnum = 54; disp('OH NO!!! SPECIAL');
end
spec_calib = load(spectrum_file);
[hsi_spec, hsi_wvl] = get_calibration_matrix(spec_calib, wmin, wmax, wnum, wscaling);

%%%GET Full-Scan results
hsi_full_scan = reconstruct_full_scan(meas.full_scan, hsi_spec, hsi_wvl);
hsi_full_scan = max(0, hsi_full_scan);
hsi_full_scan = hsi_full_scan/norm(hsi_full_scan(:));

end
