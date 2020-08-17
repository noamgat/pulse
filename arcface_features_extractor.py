import torch
import numpy as np
from skimage import transform as trans

from align_faces import get_reference_facial_points, REFERENCE_FACIAL_POINTS, FaceWarpException, \
    get_affine_transform_matrix
from config import image_h, image_w
from utils import get_central_face_attributes_img
import kornia
import cv2
from PIL import Image

class ArcfaceFeaturesExtractor(torch.nn.Module):
    def __init__(self, load_pretrained):
        super().__init__()
        if not load_pretrained:
            raise Exception("Not supported yet")
        checkpoint = 'InsightFace_v2/pretrained/BEST_checkpoint_r101.tar'
        checkpoint = torch.load(checkpoint)
        self.face_features_extractor = checkpoint['model'].module
        self.face_features_extractor.eval()
        # model = model.to(device)
        #model.eval()
        # bboxes, landmarks = get_central_face_attributes(filename)
        # img = align_face(full_path, landmarks)  # BGR
        # img = img[..., ::-1]  # RGB
        # img = transformer(img)
        #return checkpoint

    def forward(self, x: torch.Tensor):
        imgs = list(x)
        aligned_imgs = []
        for img in imgs:
            bboxes, landmarks = self.get_central_face_attributes(img)
            img = self.align_face(img, landmarks)
            img = self.transformer(img)
            aligned_imgs.append(img)
        x = torch.stack(aligned_imgs)
        features = self.face_features_extractor(x)
        return features

    def torch_img_to_numpy_img(self, img):
        numpy_img = img.cpu().numpy().transpose(1, 2, 0) * 255
        return numpy_img

    def get_central_face_attributes(self, img):
        # This does not generate gradients yet. We hope that the process works well enough without it.
        # Translate to H,W,D
        numpy_img = self.torch_img_to_numpy_img(img)

        Image.fromarray((numpy_img).astype(np.uint8)).save('debug/get_central.jpeg')

        bboxes, landmarks = get_central_face_attributes_img(numpy_img)
        return bboxes, landmarks

    def align_face(self, img, facial5points):
        #raw = cv.imread(img_fn, True)  # BGR
        facial5points = np.reshape(facial5points, (2, 5))

        crop_size = (image_h, image_w)

        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        output_size = (image_h, image_w)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size, inner_padding_factor, outer_padding, default_square)

        # dst_img = warp_and_crop_face(raw, facial5points)
        dst_img = self.warp_and_crop_face(img, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
        return dst_img
        #return img

    def transformer(self, img):
        return img

    def warp_and_crop_face(self, src_img,  # BGR
                           facial_pts,
                           reference_pts=None,
                           crop_size=(96, 112),
                           align_type='smilarity'):
        if reference_pts is None:
            if crop_size[0] == 96 and crop_size[1] == 112:
                reference_pts = REFERENCE_FACIAL_POINTS
            else:
                default_square = False
                inner_padding_factor = 0
                outer_padding = (0, 0)
                output_size = crop_size

                reference_pts = get_reference_facial_points(output_size,
                                                            inner_padding_factor,
                                                            outer_padding,
                                                            default_square)

        ref_pts = np.float32(reference_pts)
        ref_pts_shp = ref_pts.shape
        if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
            raise FaceWarpException(
                'reference_pts.shape must be (K,2) or (2,K) and K>2')

        if ref_pts_shp[0] == 2:
            ref_pts = ref_pts.T

        src_pts = np.float32(facial_pts)
        src_pts_shp = src_pts.shape
        if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
            raise FaceWarpException(
                'facial_pts.shape must be (K,2) or (2,K) and K>2')

        if src_pts_shp[0] == 2:
            src_pts = src_pts.T

        if src_pts.shape != ref_pts.shape:
            raise FaceWarpException(
                'facial_pts and reference_pts must have the same shape')

        if align_type is 'cv2_affine':
            tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
        #        print('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
        elif align_type is 'affine':
            tfm = get_affine_transform_matrix(src_pts, ref_pts)
        #        print('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
        else:
            # tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
            tform = trans.SimilarityTransform()
            tform.estimate(src_pts, ref_pts)
            tfm = tform.params[0:2, :]

        face_img = kornia.warp_affine(src_img.unsqueeze(0), torch.FloatTensor(tfm).unsqueeze(0),
                                      (crop_size[0], crop_size[1]))

        # Uncomment these lines to check equivalence of cv2.warpAffine and kornia.warp_affine
        # numpy_img = self.torch_img_to_numpy_img(src_img)
        # face_img_cv = cv2.warpAffine(numpy_img, tfm, (crop_size[0], crop_size[1]))
        # Image.fromarray(face_img_cv.astype(np.uint8)).save('debug/align_cv.jpeg')

        # numpy_img = self.torch_img_to_numpy_img(face_img[0])
        # Image.fromarray(numpy_img.astype(np.uint8)).save('debug/align_kornia.jpeg')


        return face_img  # BGR
