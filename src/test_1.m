% Input photograph
I = imread('../img/doll.png');

I = im2double(I);
IR = I(:,:,1);
IG = I(:,:,2);
IB = I(:,:,3);

% Joint image
J = imread('pencils_joint_depth.png');
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