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

function sfm2.getIsometricEgoMotion(...)
   self = {}
   xlua.unpack_class(
      self, {...}, 'sfm2.getEgoMotion', help_desc,
      {arg='im1', type='torch.Tensor', help='image 1'},
      {arg='im2', type='torch.Tensor', help='image 2'},
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
       help='Maximum distance from the epipolar line to consider a point valid in RANSAC'}
   )
   local M = torch.Tensor(4);
   local nFound, nInliers = self.im1.libsfm2.get2DEgoMotion(
      self.im1, self.im2, M, self.maxPoints, self.pointsQuality,
      self.pointsMinDistance, self.featuresBlockSize, self.trackerWinSize,
      self.trackerMaxLevel, self.ransacMaxDist, 0)
   local R = torch.FloatTensor(3,3):zero()
   R[1][1] = M[1]
   R[2][2] = M[1]
   R[1][2] = -M[2]
   R[2][1] = M[2]
   R[1][3] = M[3]
   R[2][3] = M[4]
   R[3][3] = 1
   return sfm2.inverse(R), nFound, nInliers;
end

function sfm2.getIsometricEgoMotion_testme()
   local im1 = image.lena()
   local im2 = torch.Tensor(im1:size())
   im2:sub(1,3,1,im2:size(2), 51, im2:size(3)):copy(image.rotate(im1, -0.1):sub(1,3,1,im2:size(2), 1, im2:size(3)-50))
   --image.display{im1, im2}
   local R, nFound, nInliers = sfm2.getIsometricEgoMotion{im1=im1, im2=im2, ransacMaxDist=10}
   print(R)
   print(nFound, nInliers)
   local K = torch.FloatTensor(3,3):copy(torch.eye(3))
   local im3, mask = sfm2.removeEgoMotion(im1, K, R, 'bilinear')
   im3[1]:cmul(mask)
   im3[2]:cmul(mask)
   im3[3]:cmul(mask)
   image.display{image={im1, im2, im3, im2-im3}, zoom=0.5}
   image.display{image=mask, zoom=0.5}
   print(M)
end

function sfm2.getPerspectiveEgoMotion(...)
   self = {}
   xlua.unpack_class(
      self, {...}, 'sfm2.getEgoMotion', help_desc,
      {arg='im1', type='torch.Tensor', help='image 1'},
      {arg='im2', type='torch.Tensor', help='image 2'},
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
       help='Maximum distance from the epipolar line to consider a point valid in RANSAC'}
   )
   local M = torch.Tensor(3,3);
   local nFound, nInliers = self.im1.libsfm2.get2DEgoMotion(
      self.im1, self.im2, M, self.maxPoints, self.pointsQuality,
      self.pointsMinDistance, self.featuresBlockSize, self.trackerWinSize,
      self.trackerMaxLevel, self.ransacMaxDist, 1)
   local R = torch.FloatTensor(3,3):copy(M)
   R:div(R:norm())
   return sfm2.inverse(R), nFound, nInliers;
end

function sfm2.getPerspectiveEgoMotion_testme()
   local im1 = image.lena()
   local im2 = torch.Tensor(im1:size())
   im2:sub(1,3,1,im2:size(2), 51, im2:size(3)):copy(image.rotate(im1, -0.1):sub(1,3,1,im2:size(2), 1, im2:size(3)-50))
   im2:copy(image.scale(im2:sub(1,3,1, 450), im2:size(3), im2:size(2)))
   --image.display{im1, im2}
   local R, nFound, nInliers = sfm2.getPerspectiveEgoMotion{im1=im1, im2=im2, ransacMaxDist=0.02}
   print(R)
   print(nFound, nInliers)
   local K = torch.FloatTensor(3,3):copy(torch.eye(3))
   local im3, mask = sfm2.removeEgoMotion(im1, K, R, 'bilinear')
   im3[1]:cmul(mask)
   im3[2]:cmul(mask)
   im3[3]:cmul(mask)
   image.display{image={im1, im2, im3, im2-im3}, zoom=0.5}
   image.display{image=mask, zoom=0.5}
   print(M)
end

function sfm2.getPerspectiveEpipolarEgoMotion(...)
   self = {}
   xlua.unpack_class(
      self, {...}, 'sfm2.getEgoMotion', help_desc,
      {arg='im1', type='torch.Tensor', help='image 1'},
      {arg='im2', type='torch.Tensor', help='image 2'},
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
       help='Maximum distance from the epipolar line to consider a point valid in RANSAC'}
   )
   local M = torch.Tensor(3,3);
   local nFound, nInliers = self.im1.libsfm2.get2DEgoMotion(
      self.im1, self.im2, M, self.maxPoints, self.pointsQuality,
      self.pointsMinDistance, self.featuresBlockSize, self.trackerWinSize,
      self.trackerMaxLevel, self.ransacMaxDist, 2)
   local R = torch.FloatTensor(3,3):copy(M)
   R:div(R:norm())
   return sfm2.inverse(R), nFound, nInliers;
end

function sfm2.getPerspectiveEpipolarEgoMotion_testme()
   local im1 = image.lena()
   local im2 = torch.Tensor(im1:size())
   im2:sub(1,3,1,im2:size(2), 51, im2:size(3)):copy(image.rotate(im1, -0.1):sub(1,3,1,im2:size(2), 1, im2:size(3)-50))
   im2:copy(image.scale(im2:sub(1,3,1, 450), im2:size(3), im2:size(2)))
   --image.display{im1, im2}
   local R, nFound, nInliers = sfm2.getPerspectiveEpipolarEgoMotion{im1=im1, im2=im2,
								    ransacMaxDist=0.02}
   print(R)
   print(nFound, nInliers)
   local K = torch.FloatTensor(3,3):copy(torch.eye(3))
   local im3, mask = sfm2.removeEgoMotion(im1, K, R, 'bilinear')
   im3[1]:cmul(mask)
   im3[2]:cmul(mask)
   im3[3]:cmul(mask)
   image.display{image={im1, im2, im3, im2-im3}, zoom=0.5}
   image.display{image=mask, zoom=0.5}
   print(M)
end

function sfm2.getEgoMotion(...)
   local self = {}
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

function sfm2.getEgoMotion2(...)
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
   local nFound, nInliers = self.im1.libsfm2.getEgoMotion2(
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

function sfm2.getEpipoleFromMatches(matches, R, K, d)
   local e = torch.Tensor(2)
   d = d or 50
   matches.libsfm2.getEpipoleFromMatches(matches, R, K, e, d)
   return e
end

function sfm2.testme()
   print("SFM: testme...")
end
