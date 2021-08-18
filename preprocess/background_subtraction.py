import cv2 as cv
import tqdm
    
path = '/Users/johnzhou/code/dlc_videos/B125_precon_topDLC_resnet101_FearConditioningTopNov27shuffle1_950000_filtered' \
       '.mp4'
capture = cv.VideoCapture(path)
fourcc = cv.VideoWriter_fourcc(*'mp4v')

width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
fps = int(capture.get(cv.CAP_PROP_FPS))

out = cv.VideoWriter('videos/mask.mp4', fourcc, fps, frame_size)

for i in tqdm.tqdm(range(frame_count), disable=not True, desc='Calculating background'):
    
    ret, frame = capture.read()

    if not ret:
        print('Failed to read frame {}'.format(i))
        break

    print(frame.shape)

    cv.imshow('Frame', frame)
    
    # out.write(res)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

out.release()
capture.release()
cv.destroyAllWindows()
