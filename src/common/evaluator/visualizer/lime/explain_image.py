#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('../based_imagenet')
import argparse
import pickle
import pprint
import numpy
import numpy.linalg
import cv2
import sklearn.linear_model
import PIL
import PIL.Image

import chainer.serializers

import simple_classifier
import nin

import matplotlib.pyplot as plt
 

def i2a(i):
    return chr(i+ord("A"))


def a2i(a):
    return ord(a)-ord("A")


class SuperPixel:
    def __init__(self, img, region_size=20):
        self.img = numpy.array(img)
        self.sp_obj = cv2.ximgproc.createSuperpixelLSC(self.img, region_size=region_size) #int(self.img.shape[0]*self.img.shape[1]/self.target_num_sp))
        #        self.sp_obj = cv2.ximgproc.createSuperpixelLSC(self.img) #int(self.img.shape[0]*self.img.shape[1]/self.target_num_sp))
        self.sp_obj.iterate(100)
        self.labels = self.sp_obj.getLabels()
        self.subimg_for_label = {}
        for l in range(self.get_superpixel_num()):
            self.subimg_for_label[l] = self.img.copy()
            self.subimg_for_label[l][self.labels != l] = 0 

    def get_superpixel_num(self):
            return self.sp_obj.getNumberOfSuperpixels()

    def get_sub_image(self, superpixel_labels=[]):
        out_img = PIL.Image.fromarray(numpy.zeros(self.img.shape, dtype=numpy.uint8))
        for l in superpixel_labels:
            out_img += self.subimg_for_label[l]
        return out_img

    def get_edges_image(self):
        return self.sp_obj.getLabelContourMask()


class LimeImage:
    def __init__(self, img, classifier, preprocessor, empty_superpixel_prob=0.1, sigma=20):
        self.normalized_image = img
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.sigma = sigma
        self.non_zero_prob = 1-empty_superpixel_prob
        self.image_vector = preprocessor(self.normalized_image)
        self.sp_inst = SuperPixel(self.normalized_image)

    @staticmethod
    def _generate_random_bin_vectors(prob, size, dim):
        random_vector = numpy.vectorize(lambda x:1 if x<prob else 0)(numpy.random.random(size=dim*size))
        return random_vector.reshape((size, dim))

    @staticmethod
    def _extract_positive_dimensions(vector):
        return [i for i, x in enumerate(vector) if x>0.]

    @staticmethod
    def _distance(image_vec1, image_vec2):
        x = numpy.sum((image_vec1-image_vec2)**2)**0.5
        return x

    @classmethod
    def _image_diff_measure(cls, image_vec1, image_vec2, sigma):
        D = cls._distance(image_vec1, image_vec2)**2/sigma**2
        x = numpy.exp(-D)
        return x

    def generate_samples(self, num_samples):
        self.num_samples = num_samples
        self.design_matrix = self._generate_random_bin_vectors(self.non_zero_prob, num_samples, self.sp_inst.get_superpixel_num())
        self.sub_img_vecs = numpy.zeros((num_samples, 3, self.normalized_image.size[0], self.normalized_image.size[1]))
        for i, d in enumerate(self.design_matrix):
            self.sub_img_vecs[i] = self.preprocessor(self.sp_inst.get_sub_image(self._extract_positive_dimensions(d)))
        tmp = self.classifier(numpy.r_[[self.image_vector], self.sub_img_vecs])
        self.score = tmp[1:].transpose((1,0))
        self.original_score = tmp[0]

    def construct_explainer_with_label(self, label, num_patches):
        observation = self.score[label].copy()
        observation /= numpy.linalg.norm(observation)
        weights = numpy.zeros((self.num_samples))
        for i in range(self.num_samples):
            weights[i] = self._image_diff_measure(self.image_vector, self.sub_img_vecs[i], self.sigma)
        observation = weights*observation

        clf = sklearn.linear_model.Lars(n_nonzero_coefs=num_patches)
        clf.fit(self.design_matrix, observation)
        """
        clf = sklearn.linear_model.LassoCV(max_iter=10000)#(alpha=0.001)
        clf.fit(design_matrix, observation)
        pprint.pprint(clf.coef_)
        """        
        return self._extract_positive_dimensions(clf.coef_) ,clf.coef_

    def construct_explainer_image_with_label(self, lable, num_patches, edge_color=[0, 0, 255]):
        index, coef = self.construct_explainer_with_label(lable, num_patches)
        img = numpy.asarray(self.sp_inst.get_sub_image(index)).copy()
        img[self.sp_inst.get_edges_image() == -1, 0] = edge_color[0]
        img[self.sp_inst.get_edges_image() == -1, 1] = edge_color[1]
        img[self.sp_inst.get_edges_image() == -1, 2] = edge_color[2]
        return img


class Classifier:
    def __init__(self):
        model_path = '../based_imagenet/model_recent'
        print('Load model:', model_path, file=sys.stderr)
        model_obj = nin.NIN()
        chainer.serializers.load_npz(model_path, model_obj)
        self.core = simple_classifier.classifier(model_obj, None)

    def score(self, image_vecs):
        return self.core.get_score_from_vec(image_vecs)


class ImageNormalizer:
    def __init__(self, crop_size, resize_size):
        self.IMG_SIZE = crop_size  # image size by cropping
        self.IMG_SIZE2 = resize_size  # image size by resize

    def normalize(self, image):
        w, h = image.size

        top = int((w-self.IMG_SIZE)/2)
        left = int((h-self.IMG_SIZE)/2)
        image = image.crop((left, top, left+self.IMG_SIZE, top+self.IMG_SIZE)).resize((int(self.IMG_SIZE/2), int(self.IMG_SIZE/2)))
        return image


class ImagePreprocessor:
    def __init__(self, mean_image_path):
        self.mean_image = pickle.load(open(mean_image_path, 'rb'))

    def process(self, image):
        image = numpy.array(image).astype(numpy.float32).transpose(2, 0, 1)
        image -= self.mean_image
        image /= 255
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
#    parser.add_argument("--number_superpixels", "-n", help="The number of desirable total superpixels", type=int)
    parser.add_argument("--number_samples", "-N", help="The number of samples", type=int)
    parser.add_argument("--number_patches", "-K", help="The number of superpixels to explain input", type=int)
#    parser.add_argument("--label", "-l", help="The label to be explained")
    parser.add_argument("--input_image", "-i", help="path for an input image")
    parser.add_argument("--sigma", "-s", help="sigma for measure", type=float, default=0.1)

    args = parser.parse_args()

    print('Setting up example image:', args.input_image, file=sys.stderr)
    image = PIL.Image.open(args.input_image)
    print('Setting up classifier, normalizer, preprocessor objects:', file=sys.stderr)
    classifier = Classifier()
    normalizer = ImageNormalizer(352, 176)
    preprocessor = ImagePreprocessor('../based_imagenet/mean.npy')
    print('Setting up Lime:', file=sys.stderr)
    normalized_image = normalizer.normalize(image)
    lime = LimeImage(normalized_image, classifier.score, preprocessor.process)
    print('Generating and predicting samples:', file=sys.stderr)
    lime.generate_samples(args.number_samples)

    plt.figure(figsize=(16, 8))
    for label in range(19):
        print('onstructing explainer for:', i2a(label), file=sys.stderr)
        img = lime.construct_explainer_image_with_label(label, args.number_patches)
        plt.subplot(3, 7, label+1)
        plt.title('{}'.format(i2a(label)))
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    plt.show()
