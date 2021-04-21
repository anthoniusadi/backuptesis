def SSR(img,sigma):
    try:
        img = np.float64(img)+1.0
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(
            img, (0,0), sigma))
        for i in range(retinex.shape[2]):
            retinex[:, :, i] = (
                retinex[:,:,i] - np.min(retinex[:,:,i]))/(np.max(retinex[:,:,i])-np.min(retinex[:, :, i]))*255
        retinex = np.uint8(
            np.minimum(np.maximum(retinex, 0), 255))
        return retinex
    except:
        print("exit")
    return retinex
def singleScaleRetinex(img, sigma):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex    
def MSR(img, sigma_list):
    try:
        img = np.float64(img) + 1.0
        retinex = np.zeros_like(img)
        for sigma in sigma_list:
            retinex += singleScaleRetinex(img, sigma)
        retinex = retinex / len(sigma_list)
        for i in range(retinex.shape[2]):
            retinex[:,:,i] = (retinex[:,:,i] - np.min(retinex[:,:,i]))/(np.max(retinex[:,:,i])-np.min(retinex[:,:,i]))*255
        retinex = np.uint8(np.minimum(np.maximum(retinex, 0), 255))
        return retinex
    except:
        print("exit")
    return retinex
def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration
def MSRCR(img, sigma_list,  alpha, beta):
    img = np.float64(img) + 1.0
    img_retinex = MSR(img, sigma_list)    
    img_color = colorRestoration(img, alpha, beta)    
    img_msrcr = img_retinex * img_color
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:,:,i] = (img_msrcr[:,:,i] - np.min(img_msrcr[:,:,i]))/(np.max(img_msrcr[:, :, i])-np.min(img_msrcr[:,:,i]))*255
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))      
    return img_msrcr
