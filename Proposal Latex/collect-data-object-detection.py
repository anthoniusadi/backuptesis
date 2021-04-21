    if interrupt & 0xFF == 27:
        break
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'thomas/'+'thom'+str(count['tangan_thomas'])+'.jpg',img)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'meisy/'+'meis'+str(count['tangan_meisy'])+'.jpg',img)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'tongam/'+'tong'+str(count['tangan_tongam'])+'.jpg',img)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'cbnw/'+'cbn'+str(count['tangan_cbnw'])+'.jpg',img)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'aries/'+'ar'+str(count['tangan_aries'])+'.jpg',img)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'hendrik/'+'hen'+str(count['tangan_hendrik'])+'.jpg',img)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'viany/'+'via'+str(count['tangan_viany'])+'.jpg',img)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'edwin/'+'edw'+str(count['tangan_edwin'])+'.jpg',img)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'novi/'+'nov'+str(count['tangan_novi'])+'.jpg',img)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'silfany/'+'sil'+str(count['tangan_silfany'])+'.jpg',img)

