import cv2
import numpy as np


def calculate_line_deviation_from_straightness(lines):
  deviations = []
  for line in lines:
    for x1, y1, x2, y2 in line:
      length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
      if length == 0:
        continue

      mid_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
      normal = np.array([-(y2 - y1), x2 - x1])
      normal = normal / np.linalg.norm(normal)
      deviation = np.abs(np.dot(normal, mid_point - np.array([x1, y1])))
      deviations.append(deviation)

  return np.mean(deviations) if deviations else 0


def evaluate_image(image_path):
  image = cv2.imread(image_path)
  if image is None:
    print(f'Error: Unable to load image at {image_path}')
    return

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Create a mask with the same dimensions as the image, initialized to black (all zeros)
  mask = np.zeros_like(gray_image)

  # Define the strips along each edge of the image (1/4 from each margin)
  height, width = gray_image.shape
  edge_width = width // 4
  edge_height = height // 4

  # Fill the strips in the mask with white (255)
  mask[:edge_height, :] = 255  # Top strip
  mask[-edge_height:, :] = 255  # Bottom strip
  mask[:, :edge_width] = 255  # Left strip
  mask[:, -edge_width:] = 255  # Right strip

  # Apply the mask to the grayscale image
  masked_image = cv2.bitwise_and(gray_image, mask)

  # Perform edge detection using the Canny algorithm on the masked image
  edges = cv2.Canny(masked_image, 100, 200)
  edges = cv2.Canny(gray_image, 100, 200)

  # Use the Hough transform to detect lines in the edge-detected image
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

  if lines is None:
    print(f'No lines detected in {image_path}')
    return

  deviation = calculate_line_deviation_from_straightness(lines)
  print(f'Average deviation from straightness in {image_path}: {deviation}')

  # Draw the lines on the original image
  for line in lines:
    for x1, y1, x2, y2 in line:
      cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

  cv2.imwrite(f'lines_detected_{image_path}', image)


# Example usage:
image_paths = [
  'ori.jpg',
  'out.jpg',
  'geo.jpg',
  'pcn.jpg',
]
for path in image_paths:
  evaluate_image(path)
