import tensorflow as tf
import numpy as np
import cv2
import scipy.ndimage as ndi

PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

LOCAL_MAXIMUM_RADIUS = 1

POSE_CHAIN = [
    ("nose", "leftEye"), ("leftEye", "leftEar"), ("nose", "rightEye"),
    ("rightEye", "rightEar"), ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"), ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle")
]

PARENT_CHILD_TUPLES = [(PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN]

PART_CHANNELS = [
  'left_face',
  'right_face',
  'right_upper_leg_front',
  'right_lower_leg_back',
  'right_upper_leg_back',
  'left_lower_leg_front',
  'left_upper_leg_front',
  'left_upper_leg_back',
  'left_lower_leg_back',
  'right_feet',
  'right_lower_leg_front',
  'left_feet',
  'torso_front',
  'torso_back',
  'right_upper_arm_front',
  'right_upper_arm_back',
  'right_lower_arm_back',
  'left_lower_arm_front',
  'left_upper_arm_front',
  'left_upper_arm_back',
  'left_lower_arm_back',
  'right_hand',
  'right_lower_arm_front',
  'left_hand'
]

keypoint_decoder = [
    "nose",             # 0
    "leftEye",          # 1
    "rightEye",         # 2
    "leftEar",          # 3
    "rightEar",         # 4
    "leftShoulder",     # 5
    "rightShoulder",    # 6
    "leftElbow",        # 7
    "rightElbow",       # 8
    "leftWrist",        # 9
    "rightWrist",       # 10
    "leftHip",          # 11
    "rightHip",         # 12
    "leftKnee",         # 13
    "rightKnee",        # 14
    "leftAnkle",        # 15
    "rightAnkle",       # 16
]

keypoint_encoder = {x: i for i, x in enumerate(keypoint_decoder)}

# Pairs represents the lines connected from joints
# e.g. (5,6) is from leftShoulder to rightShoulder
# https://www.tensorflow.org/lite/models/pose_estimation/overview
keypoint_lines = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]
face_keypoints = [0, 1, 2, 3, 4]

# define the skeleton. code from Google's tfjs-models
# each tuple is (parent, child)
poseChain = [
  ('nose',          'leftEye'), 
  ('leftEye',       'leftEar'), 
  ('nose',          'rightEye'),
  ('rightEye',      'rightEar'), 
  ('nose',          'leftShoulder'),
  ('leftShoulder',  'leftElbow'), 
  ('leftElbow',     'leftWrist'),
  ('leftShoulder',  'leftHip'), 
  ('leftHip',       'leftKnee'),
  ('leftKnee',      'leftAnkle'), 
  ('nose',          'rightShoulder'),
  ('rightShoulder', 'rightElbow'), 
  ('rightElbow',    'rightWrist'),
  ('rightShoulder', 'rightHip'), 
  ('rightHip',      'rightKnee'),
  ('rightKnee',     'rightAnkle')
]
parentChildrenTuples = [(keypoint_encoder[parent], keypoint_encoder[child]) for (parent, child) in poseChain]
parentToChildEdges = [childId for (_, childId) in parentChildrenTuples]
childToParentEdges = [parentId for (parentId, _) in parentChildrenTuples]

class PoseNet():
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.model_fn = self.model.signatures["serving_default"]
        self.DISPLACEMENTFWD_INDEX = list(self.model_fn.structured_outputs.keys()).index('resnet_v1_50/displacement_fwd_2/BiasAdd:0')
        self.DISPLACEMENTBWD_INDEX = list(self.model_fn.structured_outputs.keys()).index('resnet_v1_50/displacement_bwd_2/BiasAdd:0')
        self.HEATMAPS_INDEX = list(self.model_fn.structured_outputs.keys()).index('float_heatmaps:0')
        self.OFFSETS_INDEX = list(self.model_fn.structured_outputs.keys()).index('float_short_offsets:0')
        self.OUTPUT_STRIDE = 16
        self.INPUT_WIDTH = self.INPUT_HEIGHT = 224
        self.MAX_POSE_DETECTION = 10
        self.MIN_SCORE = 0.15
        
    def prepare_input(self, img):
        ''' img is a (height, width, 3) image. this will resize the image to the PoseNet input dimensions,
        and add a batch dimension. Return an image with shape (1, INPUT_HEIGHT, INPUT_WIDTH, 3). Original image is not
        modified. '''
        img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        img_copy = img_copy - [123.15, 115.90, 103.06]
        img_copy = cv2.resize(img_copy, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        img_copy = np.expand_dims(img_copy, axis=0)
        img_copy = tf.convert_to_tensor(img_copy, dtype=tf.float32)
        return img_copy

    def predict(self, img):
        ''' invoke the TensorFlow Lite model. Return heatmaps, offsets, displacementFoward, and displacementBackward tensors '''
        img_copy = img

        output = self.model_fn(img_copy)
        return output.values()

    def predict_singlepose(self, img):
        ''' Wrapper around decode_singlepose. Return a list of 17 keypoints '''
        img_input = self.prepare_input(img)
        output = list(self.predict(img_input))

        heatmaps = output[self.HEATMAPS_INDEX]
        offsets = output[self.OFFSETS_INDEX]
        displacementfwd = output[self.DISPLACEMENTFWD_INDEX]
        displacementbwd = output[self.DISPLACEMENTBWD_INDEX]

        heatmaps = np.squeeze(heatmaps)
        offsets = np.squeeze(offsets)
        displacementfwd = np.squeeze(displacementfwd)
        displacementbwd = np.squeeze(displacementbwd)

        keypoints = decode_singlepose(heatmaps, offsets, self.OUTPUT_STRIDE)
        pose_scores, pose_keypoint_scores, pose_keypoint_coords = decode_multiple_poses(
            heatmaps, offsets, displacementfwd, displacementbwd, self.OUTPUT_STRIDE)

        scaleY = img.shape[0] / self.INPUT_HEIGHT
        scaleX = img.shape[1] / self.INPUT_WIDTH
        scale = np.array([scaleX, scaleY])
        for keypoint in keypoints:
            keypoint['position'] = np.round(keypoint['position'] * scale).astype(int)

        return keypoints

    def estimate_multiple_poses(self, img, max_pose_detections=10):
        img_input = self.prepare_input(img)
        output = list(self.predict(img_input))

        heatmaps = output[self.HEATMAPS_INDEX]
        offsets = output[self.OFFSETS_INDEX]
        displacementfwd = output[self.DISPLACEMENTFWD_INDEX]
        displacementbwd = output[self.DISPLACEMENTBWD_INDEX]

        heatmaps = np.squeeze(heatmaps)
        offsets = np.squeeze(offsets)
        displacementfwd = np.squeeze(displacementfwd)
        displacementbwd = np.squeeze(displacementbwd)

        pose_scores, keypoint_scores, keypoint_coords = decode_multiple_poses(
            heatmaps, offsets, displacementfwd, displacementbwd,
            output_stride=self.OUTPUT_STRIDE,
            max_pose_detections=self.MAX_POSE_DETECTION,
            min_pose_score=self.MIN_SCORE)

        scaleY = img.shape[0] / self.INPUT_HEIGHT
        scaleX = img.shape[1] / self.INPUT_WIDTH
        scale = np.array([scaleY, scaleX])

        keypoint_coords *= scale

        return pose_scores, keypoint_scores, keypoint_coords


def decode_singlepose(heatmaps, offsets, outputStride):
    ''' Decode heatmaps and offets output to keypoints. Return a list of keypoints, each keypoint is a dictionary,
    with keys 'pos' for position np.ndarray([x,y]) and 'score' for confidence score '''
    numKeypoints = heatmaps.shape[-1]

    def get_keypoint(i):
        sub_heatmap = heatmaps[:, :, i]  # heatmap corresponding to keypoint i
        y, x = np.unravel_index(np.argmax(sub_heatmap), sub_heatmap.shape)  # y, x position of the max value in heatmap
        score = sub_heatmap[y, x]  # max value in heatmap

        # convert x, y to coordinates on the input image
        y_image = y * outputStride + offsets[y, x, i]
        x_image = x * outputStride + offsets[y, x, i + numKeypoints]

        # position is wrapped in a np array to support vector operations
        pos = np.array([x_image, y_image])
        return {'position': pos, 'score': score}

    keypoints = [get_keypoint(i) for i in range(numKeypoints)]
    
    return keypoints


def within_nms_radius(poses, squared_nms_radius, point, keypoint_id):
    for _, _, pose_coord in poses:
        if np.sum((pose_coord[keypoint_id] - point) ** 2) <= squared_nms_radius:
            return True
    return False


def within_nms_radius_fast(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis=1) <= squared_nms_radius)


def get_instance_score(
        existing_poses, squared_nms_radius,
        keypoint_scores, keypoint_coords):
    not_overlapped_scores = 0.
    for keypoint_id in range(len(keypoint_scores)):
        if not within_nms_radius(
                existing_poses, squared_nms_radius,
                keypoint_coords[keypoint_id], keypoint_id):
            not_overlapped_scores += keypoint_scores[keypoint_id]
    return not_overlapped_scores / len(keypoint_scores)


def get_instance_score_fast(
        exist_pose_coords,
        squared_nms_radius,
        keypoint_scores, keypoint_coords):
    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > squared_nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])
    else:
        not_overlapped_scores = np.sum(keypoint_scores)
    return not_overlapped_scores / len(keypoint_scores)


def score_is_max_in_local_window(keypoint_id, score, hmy, hmx, local_max_radius, scores):
    height = scores.shape[0]
    width = scores.shape[1]

    y_start = max(hmy - local_max_radius, 0)
    y_end = min(hmy + local_max_radius + 1, height)
    x_start = max(hmx - local_max_radius, 0)
    x_end = min(hmx + local_max_radius + 1, width)

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if scores[y, x, keypoint_id] > score:
                return False
    return True


def build_part_with_score(score_threshold, local_max_radius, scores):
    parts = []
    height = scores.shape[0]
    width = scores.shape[1]
    num_keypoints = scores.shape[2]

    for hmy in range(height):
        for hmx in range(width):
            for keypoint_id in range(num_keypoints):
                score = scores[hmy, hmx, keypoint_id]
                if score < score_threshold:
                    continue
                if score_is_max_in_local_window(keypoint_id, score, hmy, hmx,
                                                local_max_radius, scores):
                    parts.append((
                        score, keypoint_id, np.array((hmy, hmx))
                    ))
    return parts


def build_part_with_score_fast(score_threshold, local_max_radius, scores):
    parts = []
    num_keypoints = scores.shape[2]
    lmd = 2 * local_max_radius + 1

    # NOTE it seems faster to iterate over the keypoints and perform maximum_filter
    # on each subarray vs doing the op on the full score array with size=(lmd, lmd, 1)
    for keypoint_id in range(num_keypoints):
        kp_scores = scores[:, :, keypoint_id].copy()
        kp_scores[kp_scores < score_threshold] = 0.
        max_vals = ndi.maximum_filter(kp_scores, size=lmd, mode='constant')
        max_loc = np.logical_and(kp_scores == max_vals, kp_scores > 0)
        max_loc_idx = max_loc.nonzero()
        for y, x in zip(*max_loc_idx):
            parts.append((
                scores[y, x, keypoint_id],
                keypoint_id,
                np.array((y, x))
            ))

    return parts


def decode_multiple_poses(
        scores, offsets, displacements_fwd, displacements_bwd, output_stride,
        max_pose_detections=10, score_threshold=0.5, nms_radius=20, min_pose_score=0.15):

    pose_count = 0
    pose_scores = np.zeros(max_pose_detections)
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))

    squared_nms_radius = nms_radius ** 2

    scored_parts = build_part_with_score_fast(score_threshold, LOCAL_MAXIMUM_RADIUS, scores)
    scored_parts = sorted(scored_parts, key=lambda x: x[0], reverse=True)

    # change dimensions from (h, w, x) to (h, w, x//2, 2) to allow return of complete coord array
    height = scores.shape[0]
    width = scores.shape[1]
    offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_fwd = displacements_fwd.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_bwd = displacements_bwd.reshape(height, width, 2, -1).swapaxes(2, 3)

    for root_score, root_id, root_coord in scored_parts:
        root_image_coords = root_coord * output_stride + offsets[
            root_coord[0], root_coord[1], root_id]

        if within_nms_radius_fast(
                pose_keypoint_coords[:pose_count, root_id, :], squared_nms_radius, root_image_coords):
            continue

        keypoint_scores, keypoint_coords = decode_pose(
            root_score, root_id, root_image_coords,
            scores, offsets, output_stride,
            displacements_fwd, displacements_bwd)

        pose_score = get_instance_score_fast(
            pose_keypoint_coords[:pose_count, :, :], squared_nms_radius, keypoint_scores, keypoint_coords)

        # NOTE this isn't in the original implementation, but it appears that by initially ordering by
        # part scores, and having a max # of detections, we can end up populating the returned poses with
        # lower scored poses than if we discard 'bad' ones and continue (higher pose scores can still come later).
        # Set min_pose_score to 0. to revert to original behaviour
        if min_pose_score == 0. or pose_score >= min_pose_score:
            pose_scores[pose_count] = pose_score
            pose_keypoint_scores[pose_count, :] = keypoint_scores
            pose_keypoint_coords[pose_count, :, :] = keypoint_coords
            pose_count += 1

        if pose_count >= max_pose_detections:
            break

    return pose_scores, pose_keypoint_scores, pose_keypoint_coords


def traverse_to_targ_keypoint(
        edge_id, source_keypoint, target_keypoint_id, scores, offsets, output_stride, displacements
):
    height = scores.shape[0]
    width = scores.shape[1]

    source_keypoint_indices = np.clip(
        np.round(source_keypoint / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    displaced_point = source_keypoint + displacements[
        source_keypoint_indices[0], source_keypoint_indices[1], edge_id]

    displaced_point_indices = np.clip(
        np.round(displaced_point / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    score = scores[displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    image_coord = displaced_point_indices * output_stride + offsets[
        displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    return score, image_coord


def decode_pose(
        root_score, root_id, root_image_coord,
        scores,
        offsets,
        output_stride,
        displacements_fwd,
        displacements_bwd
):
    num_parts = scores.shape[2]
    num_edges = len(PARENT_CHILD_TUPLES)

    instance_keypoint_scores = np.zeros(num_parts)
    instance_keypoint_coords = np.zeros((num_parts, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coord

    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_bwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_fwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    return instance_keypoint_scores, instance_keypoint_coords


def draw_keypoints(img, keypoints, threshold=0.5):
    ''' Draw keypoints on the given image '''
    for i, keypoint in enumerate(keypoints):
        pos = keypoint['position']
        score = keypoint['score']
        if score < threshold:
            continue    # skip if score is below threshold

        cv2.circle(img,tuple(pos),5,(0,255,0),-1)    # draw keypoint as circle
        keypoint_name = keypoint_decoder[i]
        cv2.putText(img,keypoint_name,tuple(pos),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) # put the name of keypoint

    return img


def draw_pose(img, keypoints, threshold=0.2, color=(0,255,0), keypointRadius=5, keypointThickness=-1, lineThickness=2):
    ''' Draw pose on img. keypoints is a list of 17 keypoints '''

    # draw keypoints of the face (eyes, ears and nose)
    for keypointId in face_keypoints:
        pos = keypoints[keypointId]['position'].astype(int)
        score = keypoints[keypointId]['score']
        if score < threshold:
            continue

        cv2.circle(img, tuple(pos), keypointRadius, color, keypointThickness)

    # draw lines connecting joints
    for (id1, id2) in keypoint_lines:
        pos1 = keypoints[id1]['position']
        pos2 = keypoints[id2]['position']
        score1 = keypoints[id1]['score']
        score2 = keypoints[id2]['score']

        if score1 < threshold or score2 < threshold:
            continue

        cv2.line(img, tuple(pos1), tuple(pos2), color, lineThickness)

    return img


def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0), thickness=2)
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def detect_pose(keypoints, threshold=0.1):
    result = {}

    # t-pose
    result['t-pose'] = True
    tpose_series = ['leftWrist', 'leftElbow', 'leftShoulder', 'rightShoulder', 'rightElbow', 'rightWrist']  # consider these 5 points
    tpose_series = [keypoint_encoder[x] for x in tpose_series]                                              # convert to keypoint id
    tpose_series = [keypoints[x] for x in tpose_series]                                                     # obtain positions from keypoints

    reject = False
    for tpose_point in tpose_series:
        if tpose_point['score'] < 0.2:
            reject = True
            break
    if reject:
        result['t-pose'] = False
    else:
        for i in range(len(tpose_series)-1):
            vector = tpose_series[i+1]['position'] - tpose_series[i]['position']  # get vector of consecutive keypoints
            cosAngle2 = vector[1]**2 / vector.dot(vector)       # calculate cos angle squared wrt to vertical

            if cosAngle2 > threshold:
                result['t-pose'] = False
                break

    # left-hand-up and right-hand-up
    for side in ['left', 'right']:
        key = f'{side}-hand-up'
        result[key] = True
        handup_series = ['Wrist', 'Elbow', 'Shoulder']                     # consider these 3 points
        handup_series = [keypoint_encoder[f'{side}{x}'] for x in handup_series]   # convert to keypoint id
        handup_series = [keypoints[x] for x in handup_series]                  # obtain positions

        reject = False
        for handup_point in handup_series:
            if handup_point['score'] < 0.2:
                reject = True
                break
        if reject:
            result[key] = False
        else:
            for i in range(len(handup_series)-1):
                vector = handup_series[i+1]['position'] - handup_series[i]['position']  # get vector
                if vector[1] < 0:
                    result[key] = False
                    break
                
                cosAngle = vector[0] / np.linalg.norm(vector)   # calculate cos angle wrt to horizontal

                if cosAngle > threshold:
                    result[key] = False
                    break

    return result


if __name__ == '__main__':
    # set up posenet
    model_path = 'posenet_resnet50float_stride16'
    posenet = PoseNet(model_path)

    # read image and prepare input to shape (1, height, width, 3)
    img = cv2.imread('person.jpg')

    # apply model
    keypoints = posenet.predict_singlepose(img)
    # draw keypoints on original image
    draw_keypoints(img, keypoints)
    draw_pose(img, keypoints)
    detect_pose(keypoints)

    cv2.imshow('posenet', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()