function rec_img = render_rgb_image(hsi, hsi_wvl, camspec)


rec_img = zeros(size(hsi, 1), size(hsi, 2), 3);

for kk=1:3
    resamp_wvl = interp1(camspec.wvl, camspec.resp(:, kk), hsi_wvl, 'linear', 0);
    
    tmp = reshape(hsi, [], length(hsi_wvl))*resamp_wvl(:);
    rec_img(:, :, kk) = reshape(tmp, size(hsi, 1), []);
end

end