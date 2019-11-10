I = imread('img.jpg');

if size(I,3)==3
        I = rgb2gray(I);
        imwrite(I, 'source.png');
end

blurred = arrayfun(@(z) uint8(blur_image(double(z), 100)), I);

imwrite(blurred, "blurred.png");

recovered = arrayfun(@(z) uint8(restore_image(double(z), 100, 2000, 0.02, 10)), blurred);

imwrite(recovered, "recovered.png");

disp(sum(blurred - recovered, 'all'));