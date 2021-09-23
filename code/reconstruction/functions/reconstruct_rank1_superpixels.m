function hsi_est = reconstruct_rank1_superpixels(assort_meas, assort_index, guideimg, L, num, hsi_spec, hsi_wvl)

assort_meas = double(assort_meas);
nPats = size(assort_index, 3);
wnum = length(hsi_wvl);

for zz=1:nPats
    FilCubeIndx = repmat(double(assort_index(:,:,zz)), [1 1 wnum]);
    FilCubeIndx = FilCubeIndx+1 + repmat( reshape( size(hsi_spec,1)*(0:size(hsi_spec, 2)-1), 1,1,[]), size(L, 1), size(L, 2));
    
    FilCubeVal =  hsi_spec(FilCubeIndx);
    FilCubeValVec = reshape(FilCubeVal, [], size(FilCubeVal, 3));
    FilCubeValVecCell{zz} = FilCubeValVec;
    FilCubeIndexCall{zz} = FilCubeIndx;
    
    mask_index_cell{zz} =     create_mask_pattern(double(assort_index(:,:,zz)), 20);
    mask_index_cell{zz} = 1+0*mask_index_cell{zz};
end

%%%%Reconstruction

hsi_est = 0*FilCubeValVec;

for ss=1:num
    
    idx = find(L == ss); idx = idx(:);
    
    %%%Get RGB measurements
    rgbstk = [];
    
    for ch=1:3
        tmp = guideimg(:, :, ch);
        rgbstk(:, ch) = tmp(idx);
    end
    
    [Urgb, Srgb, Vrgb] = svds(rgbstk, 1);
    
    Amat0 = [];
    ymeas0 = [];
    for zz=1:nPats
        Amat = diag(Urgb)*FilCubeValVecCell{zz}(idx, :);
        
        keep_idx = find(mask_index_cell{zz}(idx));
        Amat1 = Amat(keep_idx, :);
        Amat0 = [Amat0; Amat1];
        
        tmp = assort_meas(:,:,zz);
        ymeas1 = tmp(idx);
        
        
        ymeas1 = ymeas1(keep_idx);
        ymeas0 = [ymeas0; ymeas1];
    end
    Anorm0 = norm(Amat0);
    Amat0 = Amat0 / Anorm0;
    spec = Anorm0*(((Amat0'*Amat0 + .001*eye(size(Amat0, 2)))) \(Amat0'*ymeas0));
    hsi_est(idx, :) = Urgb*spec';
end

hsi_est = reshape(hsi_est, [size(L) wnum]);


