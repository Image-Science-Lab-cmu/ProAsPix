function hsi_est = reconstruct_rank1_superpixels_v2(assort_meas, assort_index, guideimg, L, num, hsi_spec, hsi_wvl)
fprintf('obselete\n');
keyboard

assort_index = double(assort_index);
assort_meas = double(assort_meas);
nPats = size(assort_index, 3);
wnum = length(hsi_wvl);

% for zz=1:nPats
%     mask_index_cell(:,:,zz) =     create_mask_pattern(double(assort_index(:,:,zz)), 20);
% end
% mask_index_cell(:) = 1;

%%%%Reconstruction

hsi_est = zeros([prod(size(L)) wnum]);

L1 = repmat(L,[1 1 nPats]);


guideimg_bw = mean(guideimg, 3);

for ss=1:num
    idx = find(L == ss); idx = idx(:);
    idxP = find(L1 == ss); idxP = idxP(:);
    
    %%%Get RGB measurements
%    rgbstk = [];
%    
%    for ch=1:3
%        tmp = guideimg(:, :, ch);
%        rgbstk(:, ch) = tmp(idx);
%    end
    
%    [Urgb, Srgb, Vrgb] = svds(rgbstk, 1);
    Urgb = guideimg_bw(idx);

    Amat0 = hsi_spec(1+assort_index(idxP), :);
    Amat0 = repmat(Urgb, [nPats size(Amat0, 2)]).*Amat0;
 %   Amat0 = diag(repmat(Urgb, [nPats 1]))*Amat0;
    
    ymeas0 = assort_meas(idxP);
    
 %   keep_idx = find(mask_index_cell(idxP));
 %   ymeas0 = ymeas0(keep_idx);
 %   Amat0 = Amat0(keep_idx, :);

    Anorm0 = norm(Amat0);
    Amat0 = Amat0 / Anorm0;
    
  %  spec = Anorm0*(((Amat0'*Amat0 + .002*eye(size(Amat0, 2)))) \(Amat0'*ymeas0));
    spec = (1/Anorm0)*((Amat0'*Amat0 + 0.002*eye(size(Amat0, 2)))\(Amat0'*ymeas0));
    hsi_est(idx, :) = Urgb*spec';

end


hsi_est = reshape(hsi_est, [size(L) wnum]);