import cv2 as cv
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv.VideoCapture('/Users/pitzsb1/computer-vision-practice/data/basketball_game.mp4')

if not cap.isOpened():
    print('동영상 파일을 열 수 없습니다.')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 자세 추정
    res = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.pose_landmarks:
        # 각 랜드마크 정보 콘솔 출력
        for i, landmark in enumerate(res.pose_landmarks.landmark):
            print(f"랜드마크 {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, "
                  f"z={landmark.z:.3f}, visibility={landmark.visibility:.3f}")

        # 랜드마크 시각화
        mp_drawing.draw_landmarks(
            frame,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

    cv.imshow('MediaPipe Pose - Baseball', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
pose.close()

# BlazePose는 총 33개의 신체 랜드마크를 추정하며, 각 랜드마크는 x, y, z의 3차원 좌표로 표현된다. visibility 값은 해당 랜드마크가 보일 가능성을 나타낸다.