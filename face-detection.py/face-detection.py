from pathlib import Path

import cv2


project_root = Path(__file__).resolve().parent.parent
image_path = project_root / "group-photo.webp"
output_path = project_root / "group-photo-copy.webp"
cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

group_photo = cv2.imread(str(image_path))
if group_photo is None:
    raise FileNotFoundError(f"Could not read input image: {image_path}")

face_classifier = cv2.CascadeClassifier(str(cascade_path))
if face_classifier.empty():
    raise FileNotFoundError(f"Could not load face cascade: {cascade_path}")

gray_photo = cv2.cvtColor(group_photo, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(
    gray_photo,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

for x, y, width, height in faces:
    cv2.rectangle(group_photo, (x, y), (x + width, y + height), (255, 0, 0), 2)

if not cv2.imwrite(str(output_path), gray_photo):
    raise OSError(f"Could not save grayscale image: {output_path}")

print(f"Detected {len(faces)} face(s). Grayscale image saved to {output_path}")
cv2.imshow("Group Photo", group_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()
