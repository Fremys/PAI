from tkinter import *
from PIL import ImageTk, Image, ImageEnhance

# Definindo classe para modificação da imagem
class processImage:
    def __init__(self, filename):
        self.origin_img = Image.open(filename)
        self.img = Image.open(filename)
        self.root = Tk()
        self.label = None
        self.slide = None
        
        
    def showImage(self):
        
        #padronizando o tamanho da imagem, mantendo sua proporção
        img_r = self.img.resize((400,400))
        
        # self.img = img_r
        # self.origin_img = img_r
        
        img_tk = ImageTk.PhotoImage(img_r)
        
        # self.root.geometry("1280x720")
        
        
        #criar widget de exibição
        self.label = Label(self.root, image=img_tk)
        
        # Definir função ao clique na tela
        # self.label.bind("<Button 1>", self.click_zoom)
        
        # # empacotar mudanças
        self.label.pack()
        
        #slide
        # self.slide = Scale(self.root, from_=-10, to=10, orient="horizontal", command=self.constraste)
        # self.slide.pack()
        
        self.root.mainloop()
        
    def click_zoom(self, event):
        x = event.x
        y = event.y
        
        #dar zoom
        img_origin = self.img
        img_cut = img_origin.crop((x-200, y-100, x+200, y+100))
        img = img_cut.resize((img_cut.size[0] * 2, img_cut.size[1]*2))
         
        #converter a imagem a ser mostrada
        img_tk = ImageTk.PhotoImage(img)
        
        print(img_tk)
            
        #exibir tela com o zoom
        root_z = Toplevel()
        root_z.geometry("400x200")
        label_z = Label(root_z, image=img_tk)
        label_z.pack()
        root_z.mainloop()
        
    def constraste(self, value):
        # print(value)
        self.img = ImageEnhance.Contrast(self.origin_img).enhance((int(value)))
        img_tk = ImageTk.PhotoImage(self.img)
        self.label.config(image=img_tk)
        self.label.image = img_tk

    # def zoomLevel(self, value):
        
        




# Carrega a imagem
# img = Image.open("./mamografias/DleftCC/d_left_cc (1).png")
# img = img.resize((400, 400))

zoom = processImage("./imagem/arvore.png")
zoom.showImage()



