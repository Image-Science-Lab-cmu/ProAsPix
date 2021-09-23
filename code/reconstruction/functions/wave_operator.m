function y = wave_operator(x, siz, wave, mode)

switch mode
    case 'fwd'
        x = reshape(x, [], siz(3));
        y = zeros(siz);
        for kk=1:siz(3)
            y(:, :, kk) = waverec2(x(:, kk), wave.cbook, wave.name);
        end
        
    case 'adj'
        x = reshape(x, siz);
        y = zeros(wave.siz_coeff, siz(3));
        for kk=1:siz(3)
            y(:, kk) = wavedec2(x(:,:,kk), wave.level, wave.name)';
        end
end