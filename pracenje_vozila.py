import cv2
import numpy as np
import serial
import time
import math

COM_PORT, BAUDRATE = "COM3", 115200

MARKER_LEG_CM, H_TOL = 3.0, 5

TOLERANCIJA_UGLA, MIN_VREME_OKRETANJA, GRANICA_LEFT_CM = 20, .40, 12.0

BOJE = {"r": (0, 0, 255), "g": (0, 255, 0), "z": (0, 255, 255), "p": (255, 0, 255), "n": (255, 165, 0), "b": (255, 255, 255)}

ser = None
poslednja_komanda = None
vreme_poslednje_komande = 0
poseceni_uglovi = set()
ciljni_indeks = ciljni_ugao = None
okretanje = putanja_zavrsena = False
vreme_pocetka_okretanja = 0

def povezi_serijski_port():
    try:
        veza = serial.Serial(COM_PORT, BAUDRATE, timeout=.1)
        time.sleep(2)
        print(f"Povezano na: {COM_PORT}")
        return veza
    except Exception as greska: 
        print(f"Pico W nije povezan: {greska}")
        return None

def otvori_kameru():
    kamera = cv2.VideoCapture(0)
    if kamera.isOpened(): return kamera
    print("Kamera nije pronadjena!")
    kamera.release()

def posalji_komandu(komanda):
    global poslednja_komanda, vreme_poslednje_komande
    sada = time.time()
    if komanda == poslednja_komanda and sada - vreme_poslednje_komande <= .20: return
    try:
        if ser: ser.write(f"{komanda}\n".encode())
    except Exception: pass
    print("KOMANDA:", komanda)
    poslednja_komanda, vreme_poslednje_komande = komanda, sada

def tekst(frame, vrednost, pozicija, boja="b", velicina=.55): cv2.putText(frame, vrednost, pozicija, cv2.FONT_HERSHEY_SIMPLEX, velicina, BOJE[boja], 2)
def poligon(frame, tacke, boja, debljina=2): cv2.polylines(frame, [np.asarray(tacke, dtype=np.int32)], True, BOJE[boja], debljina)
def krug(frame, centar, radijus, boja, debljina=-1): cv2.circle(frame, tuple(map(int, centar)), radijus, BOJE[boja], debljina)
def rastojanje(p1, p2): return math.dist(p1, p2)
def naziv_ugla(i): return "NEMA CILJA" if i is None else ["GORE LEVO", "GORE DESNO", "DOLE DESNO", "DOLE LEVO"][i]

def ugao(p1, p2, p3):
    v1, v2 = p1.astype(float)-p2.astype(float), p3.astype(float)-p2.astype(float)
    n = np.linalg.norm(v1)*np.linalg.norm(v2)
    return 0 if n == 0 else np.degrees(np.arccos(np.clip(np.dot(v1, v2)/n, -1, 1)))

def centar(kontura):
    m = cv2.moments(kontura)
    return None if m["m00"] == 0 else (int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"]))

def detektuj_trougao(kontura):
    povrsina, obim = cv2.contourArea(kontura), cv2.arcLength(kontura, True)
    if not 8 <= povrsina <= 5000 or obim < 8: return None
    oblik = cv2.approxPolyDP(kontura, (.08 if povrsina < 100 else .04)*obim, True)
    if len(oblik) != 3: return None
    tacke = oblik.reshape(3, 2).astype(np.float32)
    stranice = sorted(np.linalg.norm(tacke[i]-tacke[(i+1)%3]) for i in range(3))
    a, b, c = stranice
    uglovi, centar_oblika = [ugao(tacke[(i-1)%3], tacke[i], tacke[(i+1)%3]) for i in range(3)], centar(kontura)
    limit_greske, granice = (.45, (55,125)) if povrsina >= 100 else (.70, (45,135))
    if min(stranice) < 2 or abs(a*a+b*b-c*c)/max(c*c,1) > limit_greske or not granice[0] <= max(uglovi) <= granice[1] or not centar_oblika: return None
    return {"trougao": oblik, "centar": centar_oblika, "povrsina": povrsina, "uglovi": uglovi}

def detektuj_kvadrat(kontura):
    povrsina, obim = cv2.contourArea(kontura), cv2.arcLength(kontura, True)
    if not 150 <= povrsina <= 15000 or obim < 30: return None
    oblik = cv2.approxPolyDP(kontura, .04*obim, True)
    if len(oblik) != 4: oblik = cv2.approxPolyDP(kontura, .06*obim, True)
    if len(oblik) != 4: return None
    x,y,sirina,visina = cv2.boundingRect(kontura) 
    tacke = oblik.reshape(4,2).astype(np.float32)
    uglovi = [ugao(tacke[(i-1)%4],tacke[i],tacke[(i+1)%4]) for i in range(4)]
    c = centar(kontura)
    if not .55 <= sirina/float(visina) <= 1.80 or povrsina/float(sirina*visina) < .55 or any(not 55 <= u <= 125 for u in uglovi) or not c: return None
    return {"kvadrat": oblik, "centar": c, "povrsina": povrsina, "uglovi": uglovi}

def uredi_uglaste_tacke(tacke):
    tacke=np.asarray(tacke,np.float32)
    suma=tacke.sum(1)
    razlika=np.diff(tacke,axis=1).ravel()
    return np.array([tacke[np.argmin(suma)],tacke[np.argmin(razlika)],tacke[np.argmax(suma)],tacke[np.argmax(razlika)]],np.float32)
def skala(marker):
    t=marker["trougao"].reshape(3,2).astype(np.float32)
    return sum(sorted(np.linalg.norm(t[i]-t[(i+1)%3]) for i in range(3))[:2])/2/MARKER_LEG_CM
def normalizuj_ugao(u): return (u+180)%360-180
def razlika_uglova(a,b): return abs(normalizuj_ugao(a-b))
def odredi_orijentaciju(v,m):
    u=math.degrees(math.atan2(v[1]-m[1],m[0]-v[0]))
    return u, "DESNO" if -45<=u<45 else "GORE" if u<135 else "LEVO" if u>=135 or u<-135 else "DOLE"

def napravi_masku(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h=cv2.cvtColor(np.uint8([[[148,32,8]]]),cv2.COLOR_RGB2HSV)[0,0,0]
    plava,zelena,crvena=(frame[:,:,i].astype(np.int16) for i in range(3))
    crvena_maska=((crvena>70)&(crvena>zelena*1.5)&(crvena>plava*1.5)&(crvena-zelena>50)&(crvena-plava>50)).astype(np.uint8)*255
    maska=cv2.bitwise_and(cv2.inRange(hsv,np.array([max(0,h-H_TOL),130,35],np.uint8),np.array([min(179,h+H_TOL),255,220],np.uint8)),crvena_maska)
    return cv2.dilate(cv2.morphologyEx(maska,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8)),np.ones((3,3),np.uint8),iterations=1)

def pronadji_markere(frame,maska):
    trouglovi=[]
    kvadrati=[]
    for kontura in cv2.findContours(maska,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,w,h=cv2.boundingRect(kontura)
        if not(x>1 and y>1 and x+w<frame.shape[1]-1 and y+h<frame.shape[0]-1): continue
        if (t := detektuj_trougao(kontura)): trouglovi.append(t)
        if (k := detektuj_kvadrat(kontura)): kvadrati.append(k)
    return sorted(trouglovi,key=lambda m:m["povrsina"],reverse=True)[:4],kvadrati

def izaberi_markere(kandidati,prosecna_skala):
    parovi=[]
    for i,prvi in enumerate(kandidati):
        for drugi in kandidati[i+1:]:
            veliki,mali=sorted((prvi,drugi),key=lambda m:m["povrsina"],reverse=True)
            if veliki["povrsina"]/max(mali["povrsina"],1)<1.1: continue
            greska=0 if not prosecna_skala else abs(rastojanje(veliki["centar"],mali["centar"])-3*prosecna_skala)/max(3*prosecna_skala,1)
            if greska<=1: parovi.append((greska,veliki,mali))
    return min(parovi,key=lambda x:x[0])[1:] if parovi else tuple(sorted(kandidati,key=lambda m:m["povrsina"],reverse=True)) if len(kandidati)==2 else (None,None)

def prikazi_markere(frame,markeri):
    for m in markeri: 
        krug(frame,m["centar"],5,"r")
        poligon(frame,m["trougao"],"g")
    if len(markeri)!=4: return None,None
    uglovi=uredi_uglaste_tacke([m["centar"] for m in markeri])
    poligon(frame,uglovi,"r")
    return uglovi,float(np.mean([skala(m) for m in markeri]))

def upravljaj_putanjom(v,m,uglovi,prosecna_skala,trenutni):
    global ciljni_indeks,ciljni_ugao,okretanje,putanja_zavrsena,vreme_pocetka_okretanja
    if putanja_zavrsena: 
        posalji_komandu("S")
        return None
    smer=np.subtract(m,v,dtype=float) 
    smer/=np.linalg.norm(smer)
    if not okretanje:
        kandidati=[(np.linalg.norm(np.subtract(c,m)),i) for i,c in enumerate(uglovi) if i not in poseceni_uglovi and np.dot(smer,np.subtract(c,m))>0]
        ciljni_indeks=min(kandidati)[1] if kandidati else None
    if ciljni_indeks is None: 
        posalji_komandu("S")
        return None
    udaljenost=rastojanje(m,uglovi[ciljni_indeks])/prosecna_skala if prosecna_skala else None
    if okretanje:
        posalji_komandu("L")
        if time.time()-vreme_pocetka_okretanja>=MIN_VREME_OKRETANJA and razlika_uglova(trenutni,ciljni_ugao)<=TOLERANCIJA_UGLA: 
            okretanje=False
            ciljni_ugao=ciljni_indeks=None
            posalji_komandu("F")
    elif udaljenost is not None and udaljenost<=GRANICA_LEFT_CM:
        poseceni_uglovi.add(ciljni_indeks)
        posalji_komandu("S")
        if len(poseceni_uglovi)>=4: 
            putanja_zavrsena=True
            ciljni_indeks=None
        else: 
            ciljni_ugao=normalizuj_ugao(trenutni+90)
            okretanje=True
            vreme_pocetka_okretanja=time.time()
            posalji_komandu("L")
    else: 
        posalji_komandu("F")
    return udaljenost

def main():
    global ser
    ser=povezi_serijski_port()
    kamera=otvori_kameru()
    if kamera is None: return
    while True:
        uspeh,frame=kamera.read()
        if not uspeh: break
        maska=napravi_masku(frame)
        markeri,kvadrati=pronadji_markere(frame,maska)
        uglovi,prosecna_skala=prikazi_markere(frame,markeri)
        tekst(frame,"4 FIKSNA TROUGLA PRONADJENA" if len(markeri)==4 else f"TROUGLOVI: {len(markeri)}/4",(20,35),"g" if len(markeri)==4 else "r",.65)
        veliki,mali=izaberi_markere(kvadrati,prosecna_skala) if len(kvadrati)>=2 else (None,None)
        if veliki and mali and uglovi is not None:
            trenutni,smer=odredi_orijentaciju(veliki["centar"],mali["centar"])
            udaljenost=upravljaj_putanjom(veliki["centar"],mali["centar"],uglovi,prosecna_skala,trenutni)
            poligon(frame,veliki["kvadrat"],"p",3)
            poligon(frame,mali["kvadrat"],"n",3)
            krug(frame,veliki["centar"],7,"p")
            krug(frame,mali["centar"],7,"n")
            cv2.line(frame,veliki["centar"],mali["centar"],(255,255,0),3)
            tekst(frame,f"ORIJENTACIJA: {smer}",(20,310),"z",.65)
            tekst(frame,f"UGAO: {trenutni:.1f} deg",(20,340))
            tekst(frame,f"POSECENO: {len(poseceni_uglovi)}/4",(20,500))
            tekst(frame,f"MALI -> CILJ: {udaljenost:.1f} cm" if udaljenost else "",(20,400),"z")
        else: 
            posalji_komandu("S")
            tekst(frame,"VOZILO: OBA MARKERA NISU PRONADJENA",(20,70),"r",.60)
        cv2.imshow("Detekcija markera",frame)
        cv2.imshow("HSV MASKA",maska)
        if cv2.waitKey(1)&0xFF==27: break
    posalji_komandu("S") 
    kamera.release()
    cv2.destroyAllWindows()
    if ser: ser.close()
if __name__=="__main__": main()