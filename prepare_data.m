% Here are some documents.

load matrices

datadir = '~/DeepBregmanISS/data/BSR_data/train_home/' ;
files = dir (strcat (datadir, '*.jpg')) ;

num_p = 40000 ; % num of patches to be extracted
H = 321 ; W = 481 ; % image height and width
patchsize = 8 ;
M = 64 ; % num of measurements
P = 64 ; % num of features
N = 256 ; % num of atoms in dictionary

D = dict' ;
% dimension check
assert (M == size (phi, 1)) ;
assert (N == size (D, 2)) ;
assert ((P == patchsize^2) && (P == size (phi, 2)) && (P == size (D,1))) ;

% fista options
opts.pos = false ;
opts.lambda = 4.0 ;

for file = files'
    % processing image by image
    imname = file.name ;
    impath = strcat (datadir, file.name) ;
    im = imread (impath) ;
    im = double (im) ;

    % extract patches from image
    % permute and take only the first num_p patches
    num_allp = (H-patchsize+1)*(W-patchsize+1) ;
    patches = zeros (patchsize^2, num_allp) ;
    k = 1 ;
    for i=1:H-patchsize+1
        for j=1:W-patchsize+1
            p = im (i:i+patchsize-1, j:j+patchsize-1) ;
            patches (:,k) = p (:) ;
            k = k + 1 ;
        end
    end
    perm = randperm (num_allp) ;
    patches = patches (:, perm (1:num_p)) ;

    % dc removal
    F = patches - mean (patches) ;

    % fista
    X = zeros (N, num_p) ;
    parfor j=1:num_p
        f = F(:, j) ;
        x = fista_lasso (f, D, [], opts) ;
        X (:, j) = x ;
    end

    % measurements'
    Y = phi * F ;

    % check relative error of reconstuction
    rF = D * X ;
    rel_denom = norm (F) ;
    rel_err = norm (rF - F) / rel_denom ;
    if rel_err > 0.1
        fprintf (1, 'Relative error is %f\n', rel_err) ;
    end

    % save variables
    matname = strrep (imname, 'jpg', 'mat') ;
    save (matname, 'Y', 'F', 'X') ;

end

