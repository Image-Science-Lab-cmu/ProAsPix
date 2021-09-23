function [y, Amat] = wavelet_measurement_operator(x, mode, Amat, siz, meas, mPatterns, hsi_spec)

if isempty(Amat)
    for kk=1:length(mPatterns)
        assort_index = double(meas.assort_index(:,:,mPatterns(kk)));
        tmp = reshape(hsi_spec(1+assort_index(:), :), siz);
        Amat(:,:,:,kk) = tmp;
    end
end


switch mode
    case 'fwd'
        x = reshape(x, siz(1), siz(2),size(Amat, 3));
        y = zeros(siz(1), siz(2), size(Amat, 4));
        for kk=1:size(Amat, 4)
            y(:,:,kk) = sum(x.*Amat(:,:,:,kk), 3);
        end
        
    case 'adj'
        x = reshape(x, siz(1), siz(2),size(Amat, 4));
        y = zeros(siz(1), siz(2), size(Amat, 3));
        
        for kk=1:size(x, 3)
            tmp = repmat(x(:,:,kk),[1 1 size(Amat, 3)]);
            y = y + tmp.*Amat(:,:,:,kk);
        end
end

end