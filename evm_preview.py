import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
import scipy.fftpack as fftpack
import scipy.signal as signal
from typing import List, Optional
import terminal as term


def getVideoFrames(
    videoFilePath: str,
    startFrameNumber: int = -1,
    endFrameNumber: int = -1,
    resize: float = 1,
):
    """
    Loading video file between startFrameNumber and endFrameNumber.
    Each frame is converted to YIQ color space before saving to output matrix.

    Parameters
    ----------
    vidoFilePath: video file path
    startFrameNumber: start frame number to read. If it is -1, then start from 0
    endFrameNumber: end frame number to read. If it is -1, then set it to the total frame number

    Returns
    -------
    fps: frame rate of the video
    frames: frames matrix with the shape (frameCount, frameHeight, frameWidth, channelCount)
    """
    frames = []
    vidcap = cv2.VideoCapture(videoFilePath)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalFrame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    if startFrameNumber == -1:
        startFrameNumber = 0
    if endFrameNumber == -1:
        endFrameNumber = totalFrame - 1
    success, image = vidcap.read()
    count = 0
    success = True
    term.Terminal.Clear()
    term.Terminal.Println("Loading video...")
    while success:
        term.Terminal.Println(f"{count}/{totalFrame-1}")

        image = cv2.resize(image, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = color.rgb2yiq(image).astype(np.float32)

        if count < startFrameNumber:
            success, image = vidcap.read()
            count += 1
            continue
        elif count >= endFrameNumber:
            break
        else:
            frames.append(image)
        success, image = vidcap.read()
        count += 1

        term.Terminal.DelteLast()

    frames = np.array(frames)

    term.Terminal.Println("Load Done")

    return fps, frames


def castFloatArray(imgIn, lowLimit=-1, highLimit=1) -> None:
    """
    Cast input float array value to between [lowLimit, highLimit]

    Parameters
    ----------
    imgIn: input image array

    Returns
    -------
    output image, value between [lowLimit, highLimit]
    """
    imgOut = imgIn.copy()
    if imgIn.dtype == np.float32 or imgIn.dtype == np.float64:
        imgOut[imgOut > highLimit] = highLimit
        imgOut[imgOut < lowLimit] = lowLimit
    return imgOut


def normFloatArray(imgIn) -> np.ndarray:
    """
    Normalize input float array value to between [0, 1]
    If the input image has multi-channels, it does the normalization
    channel by channel, independently.

    Parameters
    ----------
    imgIn: input image array

    Returns
    -------
    output image, value between [0, 1]
    """
    imgOut = imgIn.copy()
    if imgIn.max() == imgIn.min():
        imgOut = np.zeros(imgIn.shape)
    elif len(imgIn.shape) == 2:
        imgOut = (imgOut - imgOut.min()) / (imgOut.max() - imgOut.min())
    elif len(imgIn.shape) == 3:
        for c in range(3):
            imgOut[:, :, c] = (imgOut[:, :, c] - imgOut[:, :, c].min()) / (
                imgOut[:, :, c].max() - imgOut[:, :, c].min()
            )
    return imgOut.astype(np.float32)


def yiq2rgbUint(imgIn):
    imgOut = (255 * normFloatArray(color.yiq2rgb(castFloatArray(imgIn)))).astype(np.uint8)
    return imgOut


def yiq2bgrUint(imgIn):
    imgOut = (255 * normFloatArray(color.yiq2rgb(castFloatArray(imgIn)))).astype(np.uint8)
    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_RGB2BGR)
    return imgOut


def yiq2rgbFloat(imgIn):
    imgOut = normFloatArray(color.yiq2rgb(castFloatArray(imgIn)))
    return imgOut


def yiq2bgrFloat(imgIn):
    tmp = imgOut = normFloatArray(color.yiq2rgb(castFloatArray(imgIn)))
    imgOut = tmp.copy()
    imgOut[:, :, 0] = tmp[:, :, 2]
    imgOut[:, :, 1] = tmp[:, :, 1]
    imgOut[:, :, 2] = tmp[:, :, 0]
    return imgOut


def buildLaplacianPyramid(imgIn, maxLevel):
    """
    Build up to maxLevel laplacian pyramid for input image

    Parameters
    ----------
    imgIn: input image array
    maxLevel: upper limit of the pyramid levels. I also limit
           the smallest image edge in the pyramid should larger than 4 pixels

    Note
    ----------
    In the blur steps, it use a Gaussian filter with sigmaX=3, and sigmaY=3

    Returns
    -------
    Image pyramid, up to maxLevel
    """
    currentLayer = 0
    imgPyramid = []
    curSrc = imgIn.astype(np.float32)

    while currentLayer < maxLevel - 1:
        curH, curW = curSrc.shape[0:2]
        if curH < 4 or curW < 4:
            break
        currentLayer += 1
        imgBlur = cv2.GaussianBlur(curSrc, ksize=(0, 0), sigmaX=3)
        imgBlurDown = cv2.resize(imgBlur, ((curW + 1) // 2, (curH + 1) // 2))
        imgBlurUp = cv2.resize(imgBlurDown, (curW, curH))
        imgBlurUp = cv2.GaussianBlur(imgBlurUp, ksize=(0, 0), sigmaX=3)
        imgDiff = curSrc - imgBlurUp
        imgPyramid.append(imgDiff)
        curSrc = imgBlurDown

    imgPyramid.append(curSrc)
    return imgPyramid


def recreateImgsFromLapPyr(imgPyramid):
    """
    Recreate image from input Laplacian Pyramid

    Parameters
    ----------
    imgPyramid: input Laplacian Pyramid

    Note
    ----------
    In the blur step, it use a Gaussian filter with sigmaX=3, and sigmaY=3

    Returns
    -------
    Image recreated from inut Laplacian Pyramid
    """
    layerNum = len(imgPyramid)
    curSrc = imgPyramid[-1].copy()
    for i in np.arange(layerNum - 2, -1, -1):
        imgUp = cv2.resize(curSrc, (imgPyramid[i].shape[1], imgPyramid[i].shape[0]))
        imgBlurUp = cv2.GaussianBlur(imgUp, ksize=(0, 0), sigmaX=3)
        curSrc = imgBlurUp + imgPyramid[i]

    return curSrc


def buildVideoLapPyr(frames, maxLevel):
    """
    Build Laplacian pyramid for input video frames

    Parameters
    ----------
    frames: input video frames
    maxLevel: upper limit of the Laplician pyramid layers

    Returns
    -------
    Laplacian pyramid for input video frames, which is a list of
    videos that each video mapping to a layer in the laplacian pyramid.
    So each video has the shape (frameCount, videoFrameHeight, videoFrameWidth, channelCount).
    """
    pyr0 = buildLaplacianPyramid(frames[0], maxLevel)
    realMaxLevel = len(pyr0)

    term.Terminal.Println("Building Preview Pyramid...")

    resultList = []
    for i in range(realMaxLevel):
        curPyr = np.zeros([len(frames)] + list(pyr0[i].shape), dtype=np.float32)
        resultList.append(curPyr)

    for fn in range(len(frames)):

        term.Terminal.Println(f"{fn}/{len(frames)}")

        pyrOfFrame = buildLaplacianPyramid(frames[fn], maxLevel)
        for i in range(realMaxLevel):
            resultList[i][fn] = pyrOfFrame[i]

        term.Terminal.DelteLast()

    term.Terminal.Println("Build Done")

    return resultList


def recreateVideoFromLapPyr(pyrVideo):
    """
    Recreate video from input video Laplacian Pyramid

    Parameters
    ----------
    pyrVideo: input Laplacian Pyramid for video, returned by buildVideoLapPyr

    Returns
    -------
    Video recreated from inut Laplacian Pyramid
    """
    maxLevel = len(pyrVideo)
    fNumber, H, W, chNum = pyrVideo[0].shape
    videoResult = np.zeros(pyrVideo[0].shape, dtype=np.float32)
    for fn in range(videoResult.shape[0]):
        framePyr = [pyrVideo[i][fn] for i in range(maxLevel)]
        videoResult[fn] = recreateImgsFromLapPyr(framePyr)

    return videoResult


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    """
    Apply butterworth bandpass filter on input data on specified axis.

    Parameters
    ----------
    data : data to be filered
    lowcut, hightcut: the cut frequency of the bandpass filter
    fs   : sample rate
    order: The order of the filter
    axis : The axis of the input data array along which to apply the linear filter.
          The filter is applied to each subarray along this axis.

    Returns
    -------
    None
    """
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype="band")
    y = signal.lfilter(b, a, data, axis=0)
    return y


def temporal_ideal_filter(tensor, low, high, fps, axis=0):
    """
    Apply Ideal bandpass filter on input data on specified axis.

    Parameters
    ----------
    tensor : data to be filered
    low, hight: the cut frequency of the bandpass filter
    fps   : sample rate
    axis : The axis of the input data array along which to apply the linear filter.
          The filter is applied to each subarray along this axis.

    Returns
    -------
    None
    """
    fft = fftpack.fft(tensor, axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    if (bound_low == bound_high) and (bound_high < len(fft) - 1):
        bound_high += 1
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=axis)

    return np.abs(iff)


def idealFilterForVideoPyr(videoPyr, low, high, fps, roi=None):
    """
    Apply Ideal bandpass filter on input video pyramid

    Parameters
    ----------
    videoPyr : video pyramid
    low, hight: the cut frequency of the bandpass filter
    fps   : sample rate
    roi : if specified, only filter roi of frames. [TODO]Not implemented yet.

    Returns
    -------
    None
    """

    term.Terminal.Println("Ideal Filter...")

    resultVideoPyr = []
    for layer in range(len(videoPyr)):
        filteredPyr = temporal_ideal_filter(videoPyr[layer], low, high, fps, axis=0)
        resultVideoPyr.append(filteredPyr)

    return resultVideoPyr


def buttFilterForVideoPyr(videoPyr, low, high, fps, roi=None):
    """
    Apply Buttworth bandpass filter on input video pyramid

    Parameters
    ----------
    videoPyr : video pyramid
    low, hight: the cut frequency of the bandpass filter
    fps   : sample rate
    roi : if specified, only filter roi of frames. [TODO]Not implemented yet.

    Returns
    -------
    None
    """
    resultVideoPyr = []
    for layer in range(len(videoPyr)):
        filteredPyr = butter_bandpass_filter(videoPyr[layer], low, high, fps, order=1, axis=0)
        resultVideoPyr.append(filteredPyr)

    return resultVideoPyr


def amplifyTemporalColorSignal(originalPyr, filteredVideoPyr, alpha, chromAttenuation):
    """
    Amplify the input temporal color change signal

    Parameters
    ----------
    originalPyr : original video Laplacian pyramid
    filteredVideoPyr: temporal filtered video laplacian pyramid
    alpha   : amplify ratio
    chromAttenuation : for I/Q channel we can use this value for some attenuation, according to [1]

    Returns
    -------
    Amplified color change signal
    """
    amplifiedPyr = []

    for layer in range(len(filteredVideoPyr)):
        tensor = originalPyr[layer].copy()
        # if layer == len(filteredVideoPyr) - 1:
        tensor[:, :, :, 0] += filteredVideoPyr[layer][:, :, :, 0] * alpha
        tensor[:, :, :, 1] += filteredVideoPyr[layer][:, :, :, 1] * alpha * chromAttenuation
        tensor[:, :, :, 2] += filteredVideoPyr[layer][:, :, :, 2] * alpha * chromAttenuation

        amplifiedPyr.append(tensor)

    return amplifiedPyr


def amplifyTemporalMotionSignal(originalPyr, filteredVideoPyr, alpha, lambdaC, chromAttenuation):
    """
    Amplify the input temporal motion signal

    Parameters
    ----------
    originalPyr : original video Laplacian pyramid
    filteredVideoPyr: temporal filtered video laplacian pyramid
    alpha   : amplify ratio
    chromAttenuation : for I/Q channel we can use this value for some attenuation, according to [1]

    Returns
    -------
    Amplified motion signal
    """
    amplifiedPyr = []

    delta = lambdaC / 8 / (1 + alpha)
    frameCount, H, W, chCount = originalPyr[0].shape
    lamb = np.sqrt(H**2 + W**2) / 3.0

    for layer in range(len(filteredVideoPyr)):
        tensor = originalPyr[layer].copy()
        if layer == len(filteredVideoPyr) - 1 or layer == 0:
            currAlpha = 0
        elif currAlpha > alpha:
            currAlpha = alpha
        else:
            currAlpha = lamb / delta / 8 - 1

        tensor[:, :, :, 0] += filteredVideoPyr[layer][:, :, :, 0] * currAlpha
        tensor[:, :, :, 1] += filteredVideoPyr[layer][:, :, :, 1] * currAlpha * chromAttenuation
        tensor[:, :, :, 2] += filteredVideoPyr[layer][:, :, :, 2] * currAlpha * chromAttenuation

        amplifiedPyr.append(tensor)
        lamb /= 2

    return amplifiedPyr


def saveFramesToVideo(frames, videoPath):
    """
    Save frames as a video file

    Parameters
    ----------
    frames : input video frames
    videoPath: output video file

    Note:
    Here I use FOURCC("AVC1") on my macbook pro.
    If needed the FOURCC can be a parameter.

    Returns
    -------
    None
    """
    fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
    [height, width] = frames[0].shape[0:2]
    writer = cv2.VideoWriter(videoPath, fourcc, 30, (width, height), 1)
    for i in range(frames.shape[0]):
        frameBGR = yiq2bgrUint(frames[i])
        writer.write(frameBGR)
    writer.release()


def saveFramesToVideoROI(orgFrames, recreateFrames, videoPath, roi=None):
    """
    Save frames as a video file

    Parameters
    ----------
    frames : input video frames
    videoPath: output video file

    Note:
    Here I use FOURCC("AVC1") on my macbook pro.
    If needed the FOURCC can be a parameter.

    Returns
    -------
    None
    """
    fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
    [height, width] = orgFrames[0].shape[0:2]
    if roi is None:
        top = 0
        bottom = height
        left = 0
        right = width
    else:
        [top, bottom, left, right] = roi
    writer = cv2.VideoWriter(videoPath, fourcc, 30, (width, height), 1)
    for i in range(recreateFrames.shape[0]):
        recreateFramesBGR = yiq2bgrUint(recreateFrames[i])
        saveFrame = yiq2bgrUint(orgFrames[i])
        saveFrame[top:bottom, left:right] = recreateFramesBGR[top:bottom, left:right]
        writer.write(saveFrame)
    writer.release()


def emvCoreColor(frames, fps, maxLevel, freqLow, freqHigh, alpha, chromAttenuation, method="ideal"):
    """
    Core function to do Eulerian video magnification, only for color change.

    Parameters
    ----------
    frames : input video frames
    fps: frame rate of the input video frames
    maxLevel: upper limit of Laplacian Pyramid
    freqLow, freqHigh: cutoff frequency of bandpass filter
    alpha: amplification rate
    chromAttenuation: attenuation for I/Q channels
    method: "ideal" for ideal IIR bandpass filter
            "butt" for buttworth bandpass filter.

    Note:
         Actually buttworth bandpass filter is note suitable for color change application because
         it has too wide frequency response. I leave it here just for experiment and comparison.

    Returns
    -------
    Amplified video frames according to [1]
    """
    pyrVideo_ = buildVideoLapPyr(frames, maxLevel)

    term.Terminal.Println("Filtering...")

    if method == "ideal":
        filteredVideoPyr = idealFilterForVideoPyr(pyrVideo_, freqLow, freqHigh, fps)
    elif method == "butt":
        filteredVideoPyr = buttFilterForVideoPyr(pyrVideo_, freqLow, freqHigh, fps)

    term.Terminal.Println("Amplifying...")

    amplifiedPyr = amplifyTemporalColorSignal(pyrVideo_, filteredVideoPyr, alpha, chromAttenuation)
    recreateFrames = recreateVideoFromLapPyr(amplifiedPyr)

    return recreateFrames


def emvCoreMotion(
    frames,
    fps,
    maxLevel,
    freqLow,
    freqHigh,
    alpha,
    lambdaC,
    chromAttenuation,
    method="ideal",
):
    """
    Core function to do Eulerian video magnification, only for motion change.

    Parameters
    ----------
    frames : input video frames
    fps: frame rate of the input video frames
    maxLevel: upper limit of Laplacian Pyramid
    freqLow, freqHigh: cutoff frequency of bandpass filter
    alpha: amplification rate
    lambdaC: lambdaC described in [1]
    chromAttenuation: attenuation for I/Q channels
    method: "ideal" for ideal IIR bandpass filter
            "butt" for buttworth bandpass filter.

    Note:
         Actually buttworth bandpass filter is note suitable for color change application because
         it has too wide frequency response. I leave it here just for experiment and comparison.

    Returns
    -------
    Amplified video frames according to [1]
    """
    pyrVideo_ = buildVideoLapPyr(frames, maxLevel)

    if method == "ideal":
        filteredVideoPyr = idealFilterForVideoPyr(pyrVideo_, freqLow, freqHigh, fps)
    elif method == "butt":
        filteredVideoPyr = buttFilterForVideoPyr(pyrVideo_, freqLow, freqHigh, fps)

    amplifiedPyr = amplifyTemporalMotionSignal(pyrVideo_, filteredVideoPyr, alpha, lambdaC, chromAttenuation)
    recreateFrames = recreateVideoFromLapPyr(amplifiedPyr)

    return recreateFrames


def emv(
    inputVideoPath: str,
    outputVideoPath: str,
    maxLevel: int,
    freqLow: float,
    freqHigh: float,
    alpha: float,
    chromAttenuation: float,
    startFrameNumber: int,
    endFrameNumber: int,
    lambdaC: float = -1,
    app: str = "color",
    method: str = "ideal",
    roi: Optional[List[int]] = None,
):
    """
    Interface function to do Eulerian video magnification

    Parameters
    ----------
    inputVideoPath : input video path
    outputVideoPath: output video path
    maxLevel: upper limit of Laplacian Pyramid
    freqLow, freqHigh: cutoff frequency of bandpass filter
    alpha: amplification rate
    chromAttenuation: attenuation for I/Q channels
    startFrameNumber, endFrameNumber: controls the frames we need to process

    Returns
    -------
    None. The amplified video is saved to outputVideoPath
    """
    fps, frames = getVideoFrames(inputVideoPath, startFrameNumber, endFrameNumber)
    if app == "color":
        recreateFrames = emvCoreColor(
            frames, fps, maxLevel, freqLow, freqHigh, alpha, chromAttenuation, method
        )
    elif app == "motion":
        recreateFrames = emvCoreMotion(
            frames,
            fps,
            maxLevel,
            freqLow,
            freqHigh,
            alpha,
            lambdaC,
            chromAttenuation,
            method,
        )
    saveFramesToVideoROI(frames, recreateFrames, outputVideoPath, roi)
    return
