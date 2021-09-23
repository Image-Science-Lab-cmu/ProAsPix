function [hsi_stk, hsi_cor_stk] = get_single_reconstruction(meas, mPattern, supernum, hsi_spec, hsi_wvl);

%%%%%%%Super-pixel segmentation
[L, num] = superpixels(max(0, meas.guide).^(1/2.1), supernum);


assort_index = double(meas.assort_index(:,:,mPattern));
mask_index = create_mask_pattern(double(assort_index), 20);
mask_index(:) = 1;



vec = @(x) x(:);

clear hsi_est;

assort_sim = double(meas.assort_sim(:,:,mPattern))/2^16;

for zzz = 1:4
    switch zzz
        case 1
            assort_meas = double(meas.assort_sim(:,:,mPattern))/2^16;
        case 2
            assort_meas = double(meas.assort_meas(:,:,mPattern))/2^16;
        case 3
            assort_meas = double(meas.assort_restored(:,:,mPattern))/2^16;
        case 4
            assort_meas = double(meas.assort_restored_noguide(:,:,mPattern))/2^16;
    end
    
    
    hsi_est = reconstruct_rank1_superpixels_v3(assort_meas, assort_index, meas.guide, L, num, hsi_spec, hsi_wvl);
    hsi_est(isnan(hsi_est)) = 0;
    hsi_est = max(0, hsi_est);
    hsi_est = hsi_est/norm(hsi_est(:));
    
    correction = mean(meas.guide, 3)./mean(hsi_est, 3);
    hsi_est_correct = hsi_est.*repmat(correction, [1 1 size(hsi_est, 3)]);
    hsi_est_correct = hsi_est_correct/norm(hsi_est_correct(:));
    
    hsi_stk{zzz} = hsi_est;
    hsi_cor_stk{zzz} = hsi_est_correct;
end

end