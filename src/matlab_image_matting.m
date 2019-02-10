% Input photograph
clear
img_tag = 'kid2';
img_name=['../img/' img_tag '.bmp'];
scribs_img_name=['../img/' img_tag '_m.bmp'];
matte_img_name = ['../img/' img_tag '_alpha.bmp'];
res_img_name = ['../img/' img_tag '_res.bmp'];

runMatting

I = imread(img_name);

I = im2double(I);
IR = I(:,:,1);
IG = I(:,:,2);
IB = I(:,:,3);

% Joint image
J = imread(matte_img_name);
J = im2double(J);
BW = im2bw(J,0.5);

% Depth-of-field Examples
sigma_s = 10;
sigma_r = 0.2;

% Edges superimposed.
F_nc = NC(I, sigma_s, sigma_r,3,J);

% Composition
Out = F_nc;
OutR = Out(:,:,1); 
OutG = Out(:,:,2); 
OutB = Out(:,:,3); 

OutR(BW) = IR(BW);
OutG(BW) = IG(BW); 
OutB(BW) = IB(BW);

Out = cat(3,OutR,OutG,OutB);

% Show results.
figure, imshow(I); title('Input photograph');
figure, imshow(Out); title('Filtered photograph');
imwrite(Out,res_img_name)