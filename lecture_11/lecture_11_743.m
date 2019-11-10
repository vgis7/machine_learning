I = imread('img.jpg');

if size(I,3)==3
        I = rgb2gray(I);
        imwrite(I, 'source.png');
end

blurred = arrayfun(@(z) uint8(blur_image(double(z), 100)), I);

imwrite(blurred, "blurred.png");