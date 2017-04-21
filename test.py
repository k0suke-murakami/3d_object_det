from net.common import *
from net.utility.file import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from dummynet import *
from data import *

from net.rpn_loss_op import *
from net.rcnn_loss_op import *
from net.rpn_target_op import make_bases, make_anchors, rpn_target
from net.rcnn_target_op import rcnn_target

from net.rpn_nms_op     import draw_rpn_nms, draw_rpn
from net.rcnn_nms_op    import rcnn_nms, draw_rcnn_nms, draw_rcnn
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels

# output dir, etc
out_dir = '/root/sharefolder/sdcnd/didi-udacity-2017/output'
makedirs(out_dir +'/tf')
makedirs(out_dir +'/check_points')
log = Logger(out_dir+'/log.txt',mode='a')

ratios=np.array([0.5,1,2], dtype=np.float32)
scales=np.array([1,2,3],   dtype=np.float32)
bases = make_bases(
    base_size = 16,
    ratios=ratios,
    scales=scales
)
num_bases = len(bases)
stride = 8

def load_dummy_data():
    rgb   = np.load('/root/sharefolder/sdcnd/didi-udacity-2017/data/one_frame/rgb.npy')
    lidar = np.load('/root/sharefolder/sdcnd/didi-udacity-2017/data/one_frame/lidar.npy')
    top   = np.load('/root/sharefolder/sdcnd/didi-udacity-2017/data/one_frame/top.npy')
    front = np.zeros((1,1),dtype=np.float32)
    gt_labels    = np.load('/root/sharefolder/sdcnd/didi-udacity-2017/data/one_frame/gt_labels.npy')
    gt_boxes3d   = np.load('/root/sharefolder/sdcnd/didi-udacity-2017/data/one_frame/gt_boxes3d.npy')
    gt_top_boxes = np.load('/root/sharefolder/sdcnd/didi-udacity-2017/data/one_frame/gt_top_boxes.npy')

    top_image   = cv2.imread('/root/sharefolder/sdcnd/didi-udacity-2017/data/one_frame/top_image.png')
    front_image = np.zeros((1,1,3),dtype=np.float32)

    rgb =(rgb*255).astype(np.uint8)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gt_boxes3d = gt_boxes3d.reshape(-1,8,3)

    return  rgb, top, front, gt_labels, gt_boxes3d, top_image, front_image, lidar

rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, lidars = load_dummy_data()
# num_frames = len(rgbs)

top_shape   = tops[0].shape
front_shape = fronts[0].shape
rgb_shape   = rgbs[0].shape
top_feature_shape = (top_shape[0]//stride, top_shape[1]//stride)
out_shape=(8,3)


# set anchor boxes
num_class = 2 #incude background
anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
inside_inds = np.arange(0,len(anchors),dtype=np.int32)  #use all  #<todo>
print ('out_shape=%s'%str(out_shape))
# print ('num_frames=%d'%num_frames)

#load model ####################################################################################################
top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')

top_images   = tf.placeholder(shape=[None, top_shape[0], top_shape[0], top_shape[1]  ], 
                              dtype=tf.float32, name='top'  )
front_images = tf.placeholder(shape=[None, front_shape[0]], dtype=tf.float32, name='front')
rgb_images   = tf.placeholder(shape=[None, rgbs.shape[0],rgbs.shape[1], rgbs.shape[2]  ],
                              dtype=tf.float32, name='rgb'  )
top_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois'   ) #<todo> change to int32???
front_rois   = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='front_rois' )
rgb_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'   )

top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
    top_feature_net(top_images, top_anchors, top_inside_inds, num_bases)

front_features = front_feature_net(front_images)
rgb_features   = rgb_feature_net(rgb_images)

import pdb; pdb.set_trace()
fuse_scores, fuse_probs, fuse_deltas = \
    fusion_net(
        ( [top_features,     top_rois,     6,6,1./stride],
          [front_features,   front_rois,   0,0,1./stride],  #disable by 0,0
          [rgb_features,     rgb_rois,     6,6,1./stride],),
        num_class, out_shape) #<todo>  add non max suppression

