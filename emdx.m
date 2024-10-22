function imfs=emdx(data,varargin)
% EMDX - perform empirical mode decomposition on time series data
%
% Usage:
%  imfs = emdx(data[,maxorder,standdev]);
%
% Inputs:
%   data     = input data
%   maxorder = (optional) maximum order for EMD (default is 30)
%   standdev = (optional) standard deviation for sifting stopping critia (default is .5)
%
% Outputs:
%    imfs    = [modes time] matrix of intrinsic mode functions
%

% mikexcohen@gmail.com

switch nargin
    case 0
        help emdx
        error('No inputs. See help file.');
    case 1
        maxorder = 30;
        maxstd   = .5;
    case 2
        maxorder = varargin{1};
        maxstd   = .5;
    case 3
        maxorder = varargin{1};
        maxstd   = varargin{2};
    otherwise
        help emdx
        error('Too many inputs. See help file.');
end

maxiter = 1000;

%% chaek data size and adjust if necessary

npnts = length(data);
time  = 1:npnts;
% initialize
imfs  = zeros(maxorder,npnts);

%% use griddedInterpolant if exists (much faster than interp1)

% griddedInterpolat should always be preferred when your Matlab version supports it.

if exist('griddedInterpolant','file')
    dofast = true;
else dofast = false;
end


% data from this trial (must be a row vector)
imfsignal = data;

%% loop over IMF order

imforder = 1;
stop     = false;

while ~stop
    
    %% iterate over sifting process
    
    % initializations
    standdev = 10;
    numiter  = 0;
    signal   = imfsignal;
    
    % "Sifting" means iteratively identifying peaks/troughs in the
    % signal, interpolating across peaks/troughs, and then recomputing
    % peaks/troughs in the interpolated signal. Sifting ends when
    % variance between interpolated signal and previous sifting
    % iteration is minimized.
    while standdev>maxstd && numiter<maxiter
        
        % identify local min/maxima
        localmin  = [1 find(diff(sign(diff(signal)))>0)+1 npnts];
        localmax  = [1 find(diff(sign(diff(signal)))<0)+1 npnts];
        
        % create envelopes as cubic spline interpolation of min/max points
        if dofast % faster method, but works only on recent Matlab versions
            FL = griddedInterpolant(localmin(:),signal(localmin)','spline');
            FU = griddedInterpolant(localmax(:),signal(localmax)','spline');
            env_lower = FL(time);
            env_upper = FU(time);
        else % backup method, just in case
            env_lower = interp1(localmin,signal(localmin),time);
            env_upper = interp1(localmax,signal(localmax),time);
        end
        
        % compute residual and standard deviation
        prevsig   = signal;
        signal    = signal - (env_lower+env_upper)./2;
        standdev  = sum( ((prevsig-signal).^2) ./ (prevsig.^2+eps) ); % eps prevents NaN's
        
        % not too many iterations
        numiter = numiter+1;
        
    end % end sifting
    
    % imf is residual of signal and min/max average (already redefined as signal)
    imfs(imforder,:) = signal;
    imforder = imforder+1;
    
    %% residual is new signal
    
    imfsignal = imfsignal-signal;
    
    %% stop when few points are left
    
    if numel(localmax)<5 || imforder>maxorder
        stop=true;
    end
    
end % end imf for this trial

%% end
