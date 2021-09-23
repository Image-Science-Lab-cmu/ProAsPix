function hsi_est = reconstruct_rank1_superpixels_v3(assort_meas, assort_index, guideimg, L, num, hsi_spec, hsi_wvl)
assort_index = double(assort_index);
assort_meas = double(assort_meas);
nPats = size(assort_index, 3);
wnum = length(hsi_wvl);


%%%%Reconstruction

hsi_est = zeros([prod(size(L)) wnum]);

L1 = repmat(L,[1 1 nPats]);


guideimg_bw = mean(guideimg, 3);

for ss=1:num
    idx = find(L == ss); idx = idx(:);
    idxP = find(L1 == ss); idxP = idxP(:);
 
    Urgb = guideimg_bw(idx);
    
    Amat0 = hsi_spec(1+assort_index(idxP), :);
    Amat0 = repmat(Urgb, [nPats size(Amat0, 2)]).*Amat0;
    
    ymeas0 = assort_meas(idxP);
    
    Anorm0 = norm(Amat0);
    Amat0 = Amat0 / Anorm0;
    
    spec = (1/Anorm0)*((Amat0'*Amat0 + 0.002*eye(size(Amat0, 2)))\(Amat0'*ymeas0));
    hsi_est(idx, :) = Urgb*spec';

end


hsi_est = reshape(hsi_est, [size(L) wnum]);