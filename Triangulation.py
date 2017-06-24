"""
Prochaines 2H todo :
    Le buuh en divisant en gros par 2 le nombre de test à faire

Musiques :
    Jensation - Joystick

Ocr :
    Tesseract
    --> todo home

Doc :
    ocr inria : http://people.irisa.fr/Nicolas.Bechet/Publications/RNTI_SL_NB_HH_MR_vfinal.pdf
"""




import imageio 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.delaunay as triang
import pylab as py
import numpy as np
import math
import pylab
import scipy.ndimage
# 10 random points (x,y) in the plane
#x,y =  numpy.array(numpy.random.standard_normal((2,25)))
def d(x, y, xx, yy) :
        return math.sqrt((x-xx)**2 + (y-yy)**2)
        
def triangle(a) :
    xBis, yBis = transforme(a)
    xBis, yBis = np.array(xBis), np.array(yBis)
    cens,edg,tri,neig = triang.delaunay(xBis, yBis)
    
    
    taille = len(a)
    l=taille / 6
    b=0
    for t in tri:
    # t[0], t[1], t[2] are the points indexes of the triangle
        d1, d2, d3 = d(xBis[t[0]], yBis[t[0]], xBis[t[1]], yBis[t[1]]), d(xBis[t[1]], yBis[t[1]], xBis[t[2]], yBis[t[2]]), d(xBis[t[2]], yBis[t[2]], xBis[t[0]], yBis[t[0]])
        if (d1 < l and d2 < l and d3 < l) :
            t_i = [t[0], t[1], t[2], t[0]]
            if (b) :
                pylab.fill(xBis[t_i],yBis[t_i], "black")
                b=0
            else :
                b=1
        
        #pylab.plot(xBis, yBis, 'o')
        pylab.show()

#Mode rekt


jack=[]
taille = 10
for i in range(taille) :
    jack.append([])
    for j in range(taille) :
        if (i%2 == j%3) :
            jack[i].append([0, 0, 0]) #noir
        else :
            jack[i].append([1, 1, 1]) #blanc
#Ici on a les listes des coordonnées ou ya du noir

def traitElagage(jack, precision) :#Ok :)
    """Elagage <precision>"""
    taille = len(jack)
    X, Y = transforme(jack)
    X, Y = elagage(X, Y, precision)
    final = inverseTransforme(X, Y, taille)
    f(final)
def traitElagageR(jack, precision) :#Ok :)
    """Elagage <precision>"""
    taille = len(jack)
    X, Y = transforme(jack)
    X, Y = elagage(X, Y, precision)
    final = inverseTransforme(X, Y, taille)
    return final

def traitConservation(jack, i, precision) :#Ok :)
    """Conservation <i> <precision>"""
    taille = len(jack)
    X, Y = transforme(jack)
    X, Y = conservation(X, Y, i, precision)
    final = inverseTransforme(X, Y, taille)
    f(final)
def traitConservationR(jack, i, precision) :#Ok :)
    """Conservation <i> <precision>"""
    taille = len(jack)
    X, Y = transforme(jack)
    X, Y = conservation(X, Y, i, precision)
    final = inverseTransforme(X, Y, taille)
    return final

def traitSerrage(jack, rayon, loop, precision) :#Pas ok car plus on augmente dans le loop ça reduit l'image :( sinon ok :D
    """Serrage <rayon, loop, precision>"""
    taille = len(jack)
    X, Y = transforme(jack)
    print("Fin transfo")
    X, Y = serrage(X, Y, rayon, loop, precision)
    print("Fin serrage")
    final = inverseTransforme(X, Y, taille)
    f(final)
def traitSerrageR(jack, rayon, loop, precision) :#Pas ok car plus on augmente dans le loop ça reduit l'image :( sinon ok :D
    """Serrage <rayon, loop, precision>"""
    taille = len(jack)
    X, Y = transforme(jack)
    print("Fin transfo")
    X, Y = serrage(X, Y, rayon, loop, precision)
    print("Fin serrage")
    final = inverseTransforme(X, Y, taille)
    return final

def traitZoom(jack, coeff) :
    """zoom <coeff>"""
    taille = len(jack)
    X, Y = transforme(jack)
    X, Y = zoom(X, Y, coeff)
    final = inverseTransforme(X, Y, coeff*taille)
    f(final)
def traitZoomR(jack, coeff) :
    """zoom <coeff>"""
    taille = len(jack)
    X, Y = transforme(jack)
    X, Y = zoom(X, Y, coeff)
    final = inverseTransforme(X, Y, coeff*taille)
    return final

def transforme(imList) :#Ok :)
#A partir d'une liste d'une image on extrait ces Xlist et Ylist
#Ici le format choisit est une liste de liste de [0, 0, 0] ou [1, 1, 1]
    Xlist, Ylist = [], []
    for i in range(len(imList)) :
        for j in range(len(imList[i])) :
            if (imList[i][j] == [0, 0, 0]) :
                Xlist.append(i)
                Ylist.append(j)
    return Xlist, Ylist

def estDansListe(i, j, X, Y) :#Ok :)
    for ii in range(len(X)) :
        if (X[ii] == i and Y[ii] == j) :
            return True
    return False

def inverseTransforme(Xlist, Ylist, lenOriginal) :#Ok :)
#A partir de Xlist et Ylist on reconstitue l'original
    imList = []
    for i in range(lenOriginal) :
        imList.append([])
        for j in range(lenOriginal) :
            if (estDansListe(i, j, Xlist, Ylist)) : #noir
                imList[i].append([0, 0, 0])
            else :
                imList[i].append([1, 1, 1]) #blanc
    return imList
        
def remove(list, indice) :#Ok :)
    l=[]
    for i in range(len(list)) :
        if (i != indice) :
            l.append(list[i])
    return l

def dist(x1, y1, x2, y2) :#Ok :)
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def elagage (Xlist, Ylist, precision) :#Ok :)
    Xtmp = Xlist
    Ytmp = Ylist
    i=0
    while (i < len(Xtmp)) :
        j=0
        while (j < len(Ytmp)) :
            if (i != j and dist(Xtmp[i], Ytmp[i], Xtmp[j], Ytmp[j]) < precision) :
                Xtmp, Ytmp = remove(Xtmp, j), remove(Ytmp, j)
                j=-1
                i=0
            j+=1
        i+=1
    return (Xtmp, Ytmp)


def conservation (Xlist, Ylist, i, precision) :#Ok :)
    Xtmp = Xlist
    Ytmp = Ylist
    valeurTemporaireX = Xlist[i]
    valeurTemporaireY = Ylist[i]
    lentmp=len(Xtmp)
    j=0
    while (j < len(Xtmp)) :
        if (dist(valeurTemporaireX, valeurTemporaireY, Xtmp[j], Ytmp[j]) > precision) :
            Xtmp, Ytmp = remove(Xtmp, j), remove(Ytmp, j)
            j=-1
        j+=1
        
    return(Xtmp, Ytmp)

def zoom(Xlist, Ylist, coeff) :
    resX, resY = [], []
    for ii in range(len(Xlist)) :
        i, j = Xlist[ii], Ylist[ii]
        #if not coeff*i in Xlist and coeff*j in Ylist:
        for k in range(coeff) :
            for kk in range(coeff) :
                resX.append(coeff*i + k)
                resY.append(coeff*j + kk)
    return resX, resY

def barycentre(Xlist, Ylist) :#Ok :)
    n=len(Xlist)
    x=0
    y=0
    for i in range(len(Xlist)) :
        x+=Xlist[i]
        y+=Ylist[i]
    return(round(x/n), round(y/n)) 

#En gros si le barycentre est excentré par rapport au point alors
#il faut placer le point ici, sinon on elargit le champ de recherche
#et c'est grave la merde


#Sinon on supp qu'on a toute la liste en noir


def serrage(Xlist, Ylist, rayon, loop, precision) :#Ok :)
    """Barycentre des conservation<rayon> fait <loop>-fois puis elagage<precision>"""
    resX = []
    resY = []
    for j in range(loop) :
        print("Loop : ", j)
        for i in range(len(Xlist)) :
            Xtmp, Ytmp = conservation(Xlist, Ylist, i, rayon)
            a, b = barycentre(Xtmp, Ytmp)
            resX.append(a)
            resY.append(b)
        Xlist=resX
        Ylist=resY
        resX=[] 
        resY=[]
    finalX, finalY = elagage(Xlist, Ylist, precision)
    return(finalX, finalY)
#Si on param bien, en en coloriant 1/2 ça le fait
#Sinon les blancs on les recolore/découpe

def b(coeff, loop) :
    bx = [1, 1, 1, 1, 1, 2, 3, 3, 3, 2]
    by = [0, 1, 2, 3, 4, 2, 2, 3, 4, 4]
    ima = inverseTransforme(by, bx, 5)
    b = traitBisBisBisR(ima, coeff)
    bPropre = traitBisBisR(b, 3, loop, 2)
    return bPropre

"""
Trait disponible :
    +Elagage
    +Conservation
    +Serrage
    +Zoom
"""
def affichage(im, zoom) :
    tab = im.tolist()
    tab = traitSerrageR(tab, 2, 1, 1)
    tab = traitZoomR(tab, zoom)
    triangle(tab)
def Buuh(im, zoom) :
    tab = im.tolist()
    for i in range(1, 4) :
        for j in range(1, 2) :
            for k in range(1, 3) :
                tab = traitSerrageR(tab, i, j, k)
                if (zoom > 1) :
                    tab = traitZoomR(tab, zoom)
                pylab.figure("Rayon : " + str(i) + " // Loop : " + str(j) + " // Precision " + str(k))
                triangle(tab)
    
# TIPE LE FEU

#
"""im = imageio.imread("/media/eleves/SXDCFRGTH/eee.png") #ok
"""
z=2

#scipy.ndimage.interpolation.zoom(im, (2, 2, 1))

jack=[]
for i in range(35) :
    jack.append([])
    for j in range(35):
        if (i%2 == j%3) :
            jack[i].append([0, 0, 0])
        else :
            jack[i].append([1, 1, 1])

def moyenne5(a, b, c, d, e) :
    q = int((a[0] + b[0] + c[0] + d[0] + e[0])/5)
    s = int((a[1] + b[1] + c[1] + d[1] + e[1])/5)
    w = int((a[2] + b[2] + c[2] + d[2] + e[2])/5)
    f = int((a[3] + b[3] + c[3] + d[3] + e[2])/5)
    return (q, s, w, f)

def flou(im) : #tab 2D
    tmp=[]
    for ii in range(len(im)-2) :
        tmp.append([])
        for jj in range(len(im[ii])-2) :
            i=ii+1
            j=jj+1
            tmp[ii].append(moyenne5(im[i-1][j], im[i+1][j], im[i][j-1], im[i][j+1], im[i][j])) #
    return tmp

def flouIter(im, n) :
    if (n<=0) :
        return im
    else :
        return flouIter(flou(im), n-1)

def f(x) :
    plt.imshow(x)
    plt.show()

def fg(x, txt) :
    pylab.figure(txt)
    py.imshow(x,cmap=cm.gray)
    py.show()

#
"""imDepart = im
im=[] #cf transforme inverse
for i in range(len(imDepart)) :
    im.append([])
    for j in range(len(imDepart)) :
        if (i in Xlist and j in Ylist) :
            im[i].append([1, 1, 1, 1])
        else :
            im[i].append([0, 0, 0, 0])

plt.imshow(im)
plt.show()"""

#Affichage concret de texte :
"""im = imageio.imread("/media/eleves/SXDCFRGTH/211 lissé 7 for ocr.png")
blanc = imageio.imread("/media/eleves/SXDCFRGTH/blanc.png")
"""
def ecrit(s, hauteur) : #ok
    link = "Desktop/LETTRES/"
    fig = plt.figure()
    #fig.figimage(imageio.imread(link + "blanc.png"))
    alph = "abcdefghijklmnopqrstuvwxyz"
    alphabet = []
    """for i in alph :
        alphabet.append(imageio.imread("/media/eleves/SXDCFRGTH/" + i + ".png"))"""
    x=0
    y=200
    tmp = imageio.imread(link + "_a.png")
    coeff = hauteur/tmp.shape[0]
    for i in s :
        if (i == "/") :
            y-=2*hauteur
            x=0
        for j in range(len(alph)) :
            if (alph[j] == i) :
                tmp = imageio.imread(link +"_" + i + ".png")
                z=1
                z = scipy.ndimage.interpolation.zoom(tmp, 1)
                fig.figimage(tmp, 0, 0)
                x += z + 2
    fig.show()


#Retouches des lettres : on enleve le blanc inutile
"""im = imageio.imread("/media/eleves/SXDCFRGTH/e.png")
"""
def noir(pix) :#eventuellement fixer le seuil
    s = 180
    if pix[0] > s and pix[1] > s and pix[2] > s :
        return False
    return True

def retouche(im) :#ok
    tmp = im.tolist()
    ig = 0
    fin = False
    while ig < len(tmp)-1 and not fin :
        for i in tmp[ig] :
            if noir(i) :
                fin = True
                break
        ig += 1
    
    id = len(tmp)-1
    fin = False
    while id > 0 and not fin :
        for i in tmp[id] :
            if noir(i) :
                fin = True
                break
        id -= 1
    
    ih = 0
    fin = False
    while ih < len(tmp[0])-1 and not fin :
        for j in range(len(tmp)) :
            i = tmp[j][ih]
            if noir(i) :
                fin = True
                break
        ih += 1
    
    ib = len(tmp[0])-1
    fin = False
    while ib > 0 and not fin :
        for j in range(len(tmp)) :
            i = tmp[j][ib]
            if noir(i) :
                fin = True
                break
        ib -= 1
    a = im[ig-2:id+2, ih-1:ib+2, :]
    return a

def noir(pix) :#eventuellement fixer le seuil
    s = 150
    if pix[0] > s and pix[1] > s and pix[2] > s :
        return False
    return True

#"Creation de toute les lettres à partir d'une image
#imageAlphabet = imageio.imread("/media/eleves/SXDCFRGTH/alphabet.png")
def extraction(alph, im, link) : #ok
    #c'est y puis x
    seuilEspace = 2
    alphabet = alph
    tmp = im.tolist()
    lettre = 0
    xPrec = 0
    x = 0
    killStreak = 0
    #On avance jusqu'à nouveau noir :
    ok = True
    while ok :
        for y in range(len(tmp)) :
            if noir(tmp[y][x]) :
                ok = False
        x+=1
        if x >= len(tmp[0])-1 :
            return None
    while x < len(tmp[0])-1 :
        okLigne = True
        for y in range(len(tmp)) :
            if noir(tmp[y][x]) :
                okLigne = False
        if okLigne :
            killStreak += 1
        else :
            killStreak = 0
        x+=1
        if killStreak > seuilEspace : #On change de lettre
            print("ok")
            print(x)
            print(xPrec)
            #On enregiste d'abord la lettre en cours après retouche :
            imm = im[:, xPrec:x, :]
            imageio.imsave(link + alphabet[lettre]+ ".png", retouche(imm), "png")
            #On avance jusqu'à nouveau noir :
            ok = True
            while ok :
                for y in range(len(tmp)) :
                    if noir(tmp[y][x]) :
                        ok = False
                x+=1
                if x >= len(tmp[0])-1 :
                    return None
            #On recule pour rien rater
            x -= 1
            xPrec = x
            #On change le compteur de la lettre
            print(alphabet[lettre])
            lettre += 1
            #On reinitialise les compteurs
            killStreak = 0
            #Si jamais il est trop grand on arrete
            if lettre > len(alphabet) :
                return None

def seuil(image,s): 
    longueur=np.shape(image)[0]
    largeur=np.shape(image)[1]   
    image_sortie = image.copy()
    for ii in range(longueur):
        for jj in range(largeur):
            if image[ii,jj][0]>s and image[ii,jj][1]>s and image[ii,jj][2]>s:
                image_sortie[ii,jj][:]=[255, 255, 255]
            else :
                image_sortie[ii,jj][:]=[0, 0, 0]
    return(image_sortie)

def triangleImage(link, nom, s) :
    a = seuil(imageio.imread(link), s)
    a = py.rot90(a, 3)
    xBis, yBis = transforme(a.tolist())
    xBis, yBis = np.array(xBis), np.array(yBis)
    cens,edg,tri,neig = triang.delaunay(xBis, yBis)
    
    
    taille = len(a.tolist())
    l=taille / 15
    b=0
    py.figure()
    for t in tri:
    # t[0], t[1], t[2] are the points indexes of the triangle
        d1, d2, d3 = d(xBis[t[0]], yBis[t[0]], xBis[t[1]], yBis[t[1]]), d(xBis[t[1]], yBis[t[1]], xBis[t[2]], yBis[t[2]]), d(xBis[t[2]], yBis[t[2]], xBis[t[0]], yBis[t[0]])
        if (d1 < l and d2 < l and d3 < l) :
            #t_i = [t[0], t[1], t[2], t[0]]
            t_i = [t[0], t[1], t[2], t[0]]
            if (b) :
                py.fill(xBis[t_i], yBis[t_i], "black")
                b=0
            else :
                b=1
    py.savefig(nom)

def creationFinale(alph, imAlphabet, linkEnregistrement, s) :
    extraction(alph, imAlphabet, linkEnregistrement)
    for i in alph :
        link = linkEnregistrement + i + ".png"
        print(link)
        nom = linkEnregistrement + "_" + i + ".png"
        triangleImage(link, nom, s)

#creationFinale("abcdefghijklmnopqrstuvwxyz", imageAlphabet, "/media/eleves/SXDCFRGTH/LETTRES/", 150)
##Traitement avec flou post triangle (ubuntu) :


##TP TRAITEMENT IMAGES
import numpy as np
import imageio 
import pylab as py
import matplotlib.cm as cm
import matplotlib.pyplot as plt

## On charge l'image
#image_brute=imageio.imread('D:/BOULOT/PSI/Informatique/traitement_image/TP_traitement_image/vache.jpg')
image_brute = im

## affiche les trois composantes RGB : 
def troisCouleurs(image_brute) :
    maps=[cm.Reds_r,cm.Greens_r,cm.Blues_r]
    for jj in range(3):
        py.figure(jj)
        py.imshow(image_brute[:,:,jj],cmap=maps[jj])
        py.show()

## afficher en négatif : 
def affiche_negatif(image) :
    py.imshow(255-image[:,:,:])
    py.show()
    return()

## Convertion en niveau de gris

def converti_niveau_gris(image):
    return(np.uint8(0.299*image[:,:,0]+0.587*image[:,:,1]+0.114*image[:,:,2] ))

## Création du masque de lissage


def cree_masque_lissage(d):
    if d%2==0: 
        return(False)
    masque=np.ones((d,d))*1/d**2
    return(masque)
    
    ## Convolution 2D

def convolution_2D(image,masque): 
    longueur=np.shape(image)[0]
    largeur=np.shape(image)[1]
    largeur_filtre=np.shape(masque)[0]
    image_sortie=np.zeros((longueur,largeur))
    memoire = 5
    for ii in range(longueur): 
        if (ii/longueur*100 > memoire) :
            print("+", end="")
            memoire += 5
            if (memoire % 20 == 0) :
                print("//", end="")
        #print(ii/longueur*100, '%') # A décommenter si l'on souhaite avoir une idée de la progression. On pourrait mettre une waitbar.
        for jj in range(largeur):
            for kk in range((int(-(largeur_filtre-1)/2)),(int((largeur_filtre+1)/2))):
                for ll in range((int(-(largeur_filtre-1)/2)),(int((largeur_filtre+1)/2))):
                    if ii-kk>-1 and ii-kk<longueur and jj-ll>-1 and jj-ll<largeur :
                      image_sortie[ii,jj]=image_sortie[ii,jj]+image[ii-kk,jj-ll]*masque[kk+int((largeur_filtre-1)/2),ll+int((largeur_filtre-1)/2)]
    return(image_sortie)
                   

## Masque de flou Gaussien. 
def gauss(n,sigma):
    masque_gaussien=np.zeros((n,n))
    for ii in range(n):
        for jj in range(n):
            masque_gaussien[ii,jj]=1 / (sigma**2 * 2*np.pi) * np.exp(-(ii-(n-1)/2)**2/(sigma**2)-(jj-(n-1)/2)**2/(sigma**2))
    return masque_gaussien


## Réglage du contraste
def courbe_tonale_contraste(min,max,pixel):
    a=255/(max-min)
    b=-a*min
    sortie=int(a*pixel+b)
    if sortie>255 :
        sortie=255
    elif sortie<0 :
        sortie =0
    return(sortie)
## Augmenter contraste
def augmente_contraste(image,min,max) : 
    longueur=np.shape(image)[0]
    largeur=np.shape(image)[1]
    image_contrastee=imageio.Image(np.uint8(np.zeros((longueur,largeur,3))))
    for ii in range(3): 
        for jj in range(longueur): 
            for kk in range(largeur): 
                image_contrastee[jj,kk,ii]=( courbe_tonale_contraste(min,max,image[jj,kk,ii]) )
    py.imshow(image_contrastee)
    py.show()
    return(image_contrastee)
    
    
## Affichage de l'histogramme des intensité RGB. 


def affiche_hist(image) :
    couleur=['r','g','b']
    longueur=np.shape(image)[0]
    largeur=np.shape(image)[1]
    for ii in range(3): 
        tout=[] 
        for jj in range(longueur): 
            for kk in range(largeur): 
                tout.append(image[jj,kk,ii])
        py.figure(ii)    
        n, bins, patches = py.hist(tout, 256,  histtype='bar')
        py.setp(patches, 'facecolor', couleur[ii])

        plt.xlabel(' Intensité du pixel ') 
        plt.ylabel(' Nb de pixels    ')
        py.show()

## Calcule le module du filtre de Sobel. 

def module_gradient(image):
    sx=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sy=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    im_filtre_X=convolution_2D(image[:,:],sx)
    im_filtre_Y=convolution_2D(image[:,:],sy)
    image_module=np.uint8(np.sqrt(im_filtre_X**2+im_filtre_Y**2))
    py.imshow(image_module,cmap=cm.gray)
    py.show()
    return(image_module)
    

## Applique un seuil à l'image

def seuil(image,seuil): 
    longueur=np.shape(image)[0]
    largeur=np.shape(image)[1]   
    image_sortie=np.zeros((longueur,largeur)) 
    for ii in range(longueur):
        for jj in range(largeur):
            if image[ii,jj]>seuil:
                image_sortie[ii,jj]=255
            else :
                image_sortie[ii,jj]=0
    py.imshow(image_sortie,cmap=cm.gray)
    py.show()
    return()                    

## Applique un seuil à l'image puis application du négatif

def seuilNegatif(image,seuil): 
    longueur=np.shape(image)[0]
    largeur=np.shape(image)[1]   
    image_sortie=np.zeros((longueur,largeur)) 
    for ii in range(longueur):
        for jj in range(largeur):
            if image[ii,jj]>seuil:
                image_sortie[ii,jj]=0
            else :
                image_sortie[ii,jj]=255
    py.imshow(image_sortie,cmap=cm.gray)
    py.show()
    return()  

def seuilNegatifR(image,seuil): 
    longueur=np.shape(image)[0]
    largeur=np.shape(image)[1]   
    image_sortie=np.zeros((longueur,largeur)) 
    for ii in range(longueur):
        for jj in range(largeur):
            if image[ii,jj]>seuil:
                image_sortie[ii,jj]=0
            else :
                image_sortie[ii,jj]=255
    return(image_sortie)

## Applications : 

image=converti_niveau_gris(image_brute)
fg(image, "gris")
##
for i in range(5) :
    image_filtree=convolution_2D(image,cree_masque_lissage(i+1))
    fg(image_filtree, "Lissé " + str(i+1))
##

image_grad=module_gradient(image_filtree)
##

seuilNegatif(image_grad,70)


























