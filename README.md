# Level 1_ Face Recognition, Identification, and Verification
Membuat _Face Recognition, Identification, and Verification_ beserta API nya. Yang merupakan terdiri dari fitur identifikasi satu wajah, dan verifikasi 2 wajah beserta rekognisinya. Proses identifikasi menggunakan sudut(angle) untuk menentukan hasil recognisi serta verifikasi wajah.

### Requirements :
- Python
- Pytorch
- Cvzone (Face Detection)
- Flask
- opencv
- Numpy
- Resnet 18

### Preparing Dataset
Menyiapkan Folder beserta gambar wajah, yang foldernya berisikan nama-nama pemilik wajahnya  

### Embedding 
Proses embedding  dapat menggunakan [register-face.py] dengan mengganti directory folder data embeddingnya 
```sh
data_wajah = 'regis_face/'
```
dari proses ini menghasilkan return json yang akan digunakan untuk proses identifikasinya. 

### Result API 
dalam pembuatan API terdapat 3 route, yakni 
1. http://127.0.0.1:5000/regis_new_face
melakukan update id baru
**masih on going**
2. http://127.0.0.1:5000/predict
API ini melakukan proses identifikasi dan recognisi dengan inputan satu file wajah, yang akan menghasilkan Id wajah tersebut.
```sh
{
    "data": {
        "Name": "mindy_kaling"
    }
}
```
3. http://127.0.0.1:5000/compare
API ini melakukan proses verifikasi 2 wajah apakah kedua file gambar sama atau tidak, jika sama maka akan menampilkan idnya.
```sh
{
    "data": {
        "Angle": 41.349828565089965,
        "Face_Match": true,
        "Name": "mindy_kaling"
    }
}
```