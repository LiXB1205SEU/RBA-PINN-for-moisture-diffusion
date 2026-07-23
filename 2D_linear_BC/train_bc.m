function result = train_bc(varargin)
%TRAIN_BC Compatibility wrapper for the GPU paper-training entrypoint.
result = train_bc_gpu(varargin{:});
end
