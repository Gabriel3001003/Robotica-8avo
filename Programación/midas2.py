import cv2
import torch
import matplotlib.pyplot as plt

# Download the MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')  # Move the model to CPU
midas.eval()  # Set the model to evaluation mode

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Put Matplotlib in interactive mode
plt.ion()

# Create a named figure (optional)
plt.figure("Depth Map")
# Hook into OpenCV
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Transform input for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize depth map for better visualization
    output_display = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    output_display = cv2.cvtColor(output_display, cv2.COLOR_GRAY2BGR)

    # Display the results
    plt.imshow(depth_map)
    plt.draw()
    plt.pause(0.001)  # Adjust as needed

    cv2.imshow('CV2Frame', frame)
    cv2.imshow('Depth Map', output_display)  # Display depth map using OpenCV

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

plt.ioff()  # Turn off interactive mode (optional)
plt.show()