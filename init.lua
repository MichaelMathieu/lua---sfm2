require 'torch'
require 'dok'
require 'image'
require 'xlua'

local help_desc = [[
      Structure from Motion.
]]

sfm2 = {}

-- load C lib
require 'libsfm2'

function sfm2.TH2CV(im)
   return im:reshape(im:size(1),im:size(2)*im:size(3))
            :transpose(1,2):reshape(im:size(2), im:size(3), im:size(1))
end

function sfm2.CV2TH(im)
   return im:reshape(im:size(1)*im:size(2),im:size(3))
            :transpose(1,2):reshape(im:size(3), im:size(1), im:size(2))
end

function sfm2.getK(focal, imH, imW)
   local K = torch.FloatTensor(3,3):zero()
   K[1][1] = focal
   K[2][2] = focal
   K[3][3] = 1
   K[1][3] = imW/2
   K[2][3] = imH/2
   return K
end

function sfm2.inverse(M)
   local ret = torch.Tensor():typeAs(M):resizeAs(M)
   ret.libsfm2.inverseMatrix(M, ret)
   return ret
end

function sfm2.getEgoMotion(...)
   self = {}
   xlua.unpack_class(
      self, {...}, 'sfm2.getEgoMotion', help_desc,
      {arg='im1', type='torch.Tensor', help='image 1'},
      {arg='im2', type='torch.Tensor', help='image 2'},
      {arg='K', type='torch.FloatTensor', help='Calibration matrix'},
      {arg='maxPoints', type='number', help='Maximum number of tracked points', default=500},
      {arg='pointsQuality',type='number',help='Minimum quality of trackedpoints',default=0.02},
      {arg='pointsMinDistance', type='number',
       help='Minumum distance between two tracked points', default=3.0},
      {arg='featuresBlockSize', type='number',
       help='opencv GoodFeaturesToTrack block size', default=20},
      {arg='trackerWinSize', type='number',
       help='opencv calcOpticalFlowPyrLK block size', default=10},
      {arg='trackerMaxLevel', type='number',
       help='opencv GoodFeaturesToTrack pyramid depth', default=5},
      {arg='ransacMaxDist', type='number', default = 0.2,
       help='Maximum distance from the epipolar line to consider a point valid in RANSAC'},
      {arg='getInliers', type='bool', default = false,
       help = 'Returns the RANSAC inliers'}
   )
   local R = torch.FloatTensor(3,3)
   local T = torch.FloatTensor(3)
   local fundmat = torch.FloatTensor(3,3)
   local inliers = torch.Tensor()
   if self.getInliers then
      inliers:resize(self.maxPoints, 4)
   end
   local nFound, nInliers = self.im1.libsfm2.getEgoMotion(
      self.im1, self.im2, self.K, R, T, fundmat, inliers, self.maxPoints, self.pointsQuality,
      self.pointsMinDistance, self.featuresBlockSize, self.trackerWinSize,
      self.trackerMaxLevel, self.ransacMaxDist)
   if self.getInliers then
      return R, T, nFound, nInliers, fundmat, inliers:resize(nInliers, 4)
   else
      return R, T, nFound, nInliers, fundmat
   end
end

function sfm2.removeEgoMotion(im, K, R, mode)
   local ret = torch.Tensor(im:size()):zero()
   local mask = torch.Tensor(im:size(2), im:size(3)):zero()
   mode = mode or 'simple'
   local bilinear
   if mode == 'bilinear' then
      bilinear = true
   elseif mode == 'simple' then
      bilinear = false
   else
      error('Unknown mode ' .. mode .. ' (use : simple | bilinear)')
   end
   im.libsfm2.removeEgoMotion(im, K, R, ret, mask, bilinear)
   return ret, mask
end

function sfm2.chessboardCalibrate(images, pattern_rows, pattern_cols)
   if type(images) ~= 'table' then
      error('sfm2.chessboardCalibrate: images must be a table of images')
   end
   if #images < 3 then
      error('sfh2.chessboardCalibrate: calibration is impossible with less than 3 images')
   end
   local K = torch.Tensor(3, 3)
   local distortion = torch.Tensor(5)
   images[1].libsfm2.chessboardCalibrate(images, pattern_rows, pattern_cols, K, distortion)
   return K, distortion
end

function sfm2.undistortImage(im, K, distortParameters)
   local imt = sfm2.TH2CV(im)
   local ret = torch.Tensor(imt:size())
   local K_, dist_
   if (K:type() ~= torch.Tensor():type()) then
      K_ = torch.Tensor():resize(K:size()):copy(K)
      dist_ = torch.Tensor():resize(distortParameters:size()):copy(distortParameters)
   else
      K_ = K
      dist_ = distortParameters
   end
   im.libsfm2.undistortImage(imt, K_, dist_, ret)
   return sfm2.CV2TH(ret)
end

function sfm2.getEpipoles(fundmat)
   local e1 = torch.Tensor(2)
   local e2 = torch.Tensor(2)
   fundmat.libsfm2.getEpipoles(fundmat, e1, e2)
   return e1, e2
end

function sfm2.testme()
   print("SFM: testme...")
end
