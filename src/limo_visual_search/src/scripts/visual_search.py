#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LimoVisualSearch:
    def __init__(self):
        rospy.init_node("limo_visual_search")

        self.bridge = CvBridge()
        self.found = False

        # ===============================
        # Load target image
        # ===============================
        pkg_path = os.path.dirname(os.path.realpath(__file__))
        target_path = os.path.join(pkg_path, "../images/cereal.jpg")

        self.target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        if self.target_img is None:
            rospy.logerr("Target image not found!")
            return

        # ===============================
        # SIFT detector
        # ===============================
        self.sift = cv2.SIFT_create()
        self.kp_target, self.des_target = self.sift.detectAndCompute(
            self.target_img, None
        )

        if self.des_target is None:
            rospy.logerr("No features found in target image!")
            return

        # ===============================
        # KNN Matcher
        # ===============================
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        # ===============================
        # Subscribe to camera
        # ===============================
        rospy.Subscriber(
            "/camera_ir/color/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        rospy.loginfo("Visual search node started (SIFT + KNN)")

    def image_callback(self, msg):
        if self.found:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_frame, des_frame = self.sift.detectAndCompute(gray, None)
        if des_frame is None:
            return

        matches = self.matcher.knnMatch(self.des_target, des_frame, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  
                good_matches.append(m)

        MIN_MATCH_COUNT = 8
        if len(good_matches) < MIN_MATCH_COUNT:
            rospy.loginfo_throttle(2, f"Searching... matches = {len(good_matches)}")
            cv2.imshow("Camera View", frame)
            cv2.waitKey(1)
            return

        # ===============================
        # HOMOGRAPHY CHECK
        # ===============================
        src_pts = np.float32(
            [self.kp_target[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp_frame[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return

        inliers = int(mask.sum())
        if inliers < 15:
            return  # ❌ reject walls

        # ===============================
        # SPATIAL COMPACTNESS CHECK
        # ===============================
        pts = dst_pts[mask.ravel() == 1]
        hull = cv2.convexHull(pts.astype(np.int32))
        area = cv2.contourArea(hull)

        if area < 4000:  # walls usually spread out
            return

        # ===============================
        # ✅ CONFIRMED DETECTION
        # ===============================
        self.found = True
        rospy.loginfo("✅ CEREAL BOX FOUND (validated)")

        match_img = cv2.drawMatches(
            self.target_img,
            self.kp_target,
            gray,
            kp_frame,
            good_matches,
            None,
            matchesMask=mask.ravel().tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imshow("WHY FOUND (Matches)", match_img)
        cv2.waitKey(1)



if __name__ == "__main__":
    try:
        LimoVisualSearch()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
