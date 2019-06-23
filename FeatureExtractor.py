import cv2
import dlib
from imutils import face_utils
import numpy as np
from os.path import join


def largest_box(arr: np.ndarray) -> np.ndarray:
    index = np.argmax(arr, axis=0)[2]
    return arr[index]


class FeatureExtractor(object):
    face_cascade = cv2.CascadeClassifier(join('extraction_models', 'haarcascade_frontalface_default.xml'))
    face_detector = dlib.get_frontal_face_detector()
    # face_network = cv2.dnn.readNetFromCaffe(join('extraction_models', 'deploy.prototxt'),
    #                                         join('extraction_models', 'res10_300x300_ssd_iter_140000.caffemodel'))
    landmark_predictor = dlib.shape_predictor(join('extraction_models', 'shape_predictor_68_face_landmarks.dat'))

    def __init__(self, video_file: str):
        if video_file is '':
            return
        self.video = cv2.VideoCapture(video_file)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.frames = np.empty(0)
        self.frames_gray = np.empty(0)
        self.faces = np.empty(0)
        self.mouths = []
        self.features = np.empty(0)
        self.lips = np.empty(0)

    @classmethod
    def from_image(cls, image_file: str):
        fe = cls('')
        fe.video = None
        fe.frames = [cv2.resize(cv2.imread(image_file), None, fx=0.33, fy=0.33)]
        fe.frames_gray = [cv2.cvtColor(fe.frames[0], cv2.COLOR_BGR2GRAY)]
        fe.length = 1
        (fe.height, fe.width, _) = fe.frames[0].shape

        fe.faces = np.empty(0)
        fe.features = np.empty(0)
        fe.lips = np.empty(0)

        return fe

    def read_video(self):
        self.frames = np.empty((self.length, self.height, self.width, 3), dtype=np.uint8)
        self.frames_gray = np.empty((self.length, self.height, self.width), dtype=np.uint8)
        # self.frames_gray = []
        for i in range(self.length):
            _, frame = self.video.read()
            if frame is None:
                break
            self.frames[i] = frame
            self.frames_gray[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # self.frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def face_detect(self, draw: bool = False):
        self.faces = np.empty((self.length, 4), dtype=int)
        for i in range(len(self.frames)):
            frame_gray = self.frames_gray[i]
            # faces = FeatureExtractor.face_cascade.detectMultiScale(frame_gray)
            # if len(faces) > 0:
            #     self.faces[i] = largest_box(faces)
            #     if draw:
            #         (x, y, w, h) = largest_box(faces)
            #         cv2.rectangle(frame_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # else:  # face not found, error in detection, ignore frame
            #     self.faces[i] = 0
            face_rects = FeatureExtractor.face_detector(frame_gray, 0)
            if len(face_rects) > 0:
                self.faces[i] = face_utils.rect_to_bb(face_rects[0])
                if draw:
                    (x, y, w, h) = face_utils.rect_to_bb(face_rects[0])
                    cv2.rectangle(frame_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # frame = self.frames[i]
            # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            # FeatureExtractor.face_network.setInput(blob)
            # box = (FeatureExtractor.face_network.forward()[0, 0, 0, 3:7]
            #        * np.array([self.width, self.height, self.width, self.height])).astype(int)
            # self.faces[i] = box[0], box[1], box[2] - box[0], box[3] - box[1]
            # if draw:
            #     cv2.rectangle(frame_gray, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    def landmark_detect(self, draw: bool = False):
        self.features = np.empty((self.length, 4, 2), dtype=int)
        for i in range(len(self.frames)):
            face = self.faces[i]
            if not (face == 0).all():
                rect = dlib.rectangle(face[0], face[1], face[0] + face[2], face[1] + face[3])
                shape = self.landmark_predictor(self.frames_gray[i], rect)
                shape = face_utils.shape_to_np(shape)
                if draw:
                    for (x, y) in shape:
                        cv2.circle(self.frames_gray[i], (x, y), 1, (0, 0, 255), -1)
                self.features[i] = np.array((shape[51], shape[57], shape[48], shape[54]))
            else:
                self.features[i] = 0

    def crop_lips(self):
        self.lips = np.empty((self.length, 60, 80), dtype=np.uint8)
        center_x = 100
        center_y = 100
        for i in range(len(self.frames)):
            if not (self.features[i] == 0).all():
                center_x = int((self.features[i][2][0] + self.features[i][3][0]) / 2)
                center_y = int((self.features[i][0][1] + self.features[i][1][1]) / 2)
            else:
                self.lips = None
                return
            self.lips[i] = self.frames_gray[i][center_y - 30: center_y + 30, center_x - 40: center_x + 40]

    def show_video(self):
        for frame in self.frames_gray:
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
