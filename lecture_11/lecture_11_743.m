disp("Opening Image");
I = imread('img.jpg');

if size(I,3)==3
    disp("Converting to greyscale");
    I = rgb2gray(I);
    imwrite(I, 'source.png');
end

disp("Blurring");
%blurred = arrayfun(@(z) uint8(blur_image(double(z), 100)), I);

disp("Saving blurred.png");
imwrite(blurred, "blurred.png");


disp("recovering");
recovered = restore_image(blurred, 100, 2000, 0.02, 10);

disp("Saving recovered.png");
imwrite(recovered, "recovered.png");

disp(sum(blurred - recovered, 'all'));