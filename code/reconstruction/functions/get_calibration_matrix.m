function [A2, A2_lambda] = get_calibration_matrix(spec_calib, wmin, wmax, wnum, wscaling)


A0_lambda = spec_calib.wvl;
A0 = spec_calib.spectra_norm;
%restrict wavelength range
idx = find( (A0_lambda >= wmin) & (A0_lambda <=wmax));
A1_lambda = A0_lambda(idx);
A1 = A0(:, idx);

%reduce wavelength resolution
switch wscaling
    case 'special'
        wspacing = (wmax-wmin)/54;
        wmid = wmin:10:wmax;
        
        A2 = zeros(size(A1, 1), wnum);
        A2_lambda = zeros(wnum, 1);
        
        for ee=1:wnum
            [~, idx] = find( abs(A1_lambda-wmid(ee)) <= wspacing/2);
            A2_lambda(ee) = mean(A1_lambda(idx));
            A2(:, ee) = mean(A1(:, idx), 2);
            
        end
        
    case 'linear'
        wspacing = (wmax - wmin)/wnum;
        wmid = wmin+wspacing/2+(0:wnum-1)*wspacing;
        
        A2 = zeros(size(A1, 1), wnum);
        A2_lambda = zeros(wnum, 1);
        
        for ee=1:wnum
            [~, idx] = find( abs(A1_lambda-wmid(ee)) <= wspacing/2);
            A2_lambda(ee) = mean(A1_lambda(idx));
            A2(:, ee) = mean(A1(:, idx), 2);
            
        end
        
        
    case '1/linear'
        
        A1_kappa = 1./A1_lambda;
        kspacing = (1/wmin - 1/wmax)/wnum;
        kmid = (1/wmax)+kspacing/2+(0:wnum-1)*kspacing;
        
        A2 = zeros(size(A1, 1), wnum);
        A2_lambda = zeros(wnum, 1);
        
        for ee=1:wnum
            [~, idx] = find( abs(A1_kappa-kmid(ee)) <= kspacing/2);
            A2_kappa(ee) = mean(A1_kappa(idx));
            A2(:, ee) = mean(A1(:, idx), 2);
            A2_lambda(ee) = 1./A2_kappa(ee);
        end
        
end
