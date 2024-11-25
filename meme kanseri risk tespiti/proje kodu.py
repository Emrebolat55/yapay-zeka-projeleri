from sklearn.ensemble import RandomForestClassifier

# Eğitim veri seti
X_train = [
    [10, 0, 0, 0, 15, 15],   # Yaş, Ailesel meme kanseri öyküsü, Kişisel meme kanseri öyküsü, Çocuk doğurma yaşı, Menstrual öykü, Beden yapısı
    [30, 50, 0, 0, 25, 25],
    # Diğer eğitim verileri buraya eklenebilir
]
y_train = ["Düşük", "Yüksek"]  # Eğitim veri setine karşılık gelen sonuçlar

# Modeli eğitme
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Soru ve puan aralıkları
questions = {
    "Yaş": {20: 10, 30: 20, 40: 30, 50: 40},  # Yaş için puan eklendi
    "Ailesel meme kanseri öyküsü": {"yok": 0, "bir hala/teyze ya da büyükanne": 50, "anne ya da kız kardeş": 100, "anne ve kız kardeş": 150, "anne ve iki kız kardeş": 200},
    "Kişisel meme kanseri öyküsü": {"meme kanseri yok": 0, "meme kanseri var": 300},
    "Çocuk doğurma yaşı": {"otuz yaş öncesi ilk doğum": 0, "otuz yaş sonrası ilk doğum": 25, "çocuk yok": 50},
    "Menstrual öykü": {"menstrurasyon başlama yaşı 15 ve üstü": 15, "menstrurasyon başlama yaşı 12-14": 25, "menstrurasyon başlama yaşı 11 ve altı": 50},
    "Beden yapısı": {"zayıf": 15, "orta": 25, "şişman": 50}
}

# Yeni bir veri noktası üzerinde tahmin yapma
def predict_cancer_risk(answers):
    new_data = []
    for question, answer in answers.items():
        if question != "Yaş":
            new_data.append(questions[question][answer])
        else:
            if answer <= 0:
                raise ValueError("Yaş değeri 0'dan büyük olmalıdır.")
            new_data.append(questions[question][answer])
    prediction = model.predict([new_data])
    return prediction[0]

# Risk düzeyini belirleme
def determine_risk_level(score):
    if score <= 200:
        return "Düşük risk"
    elif 201 <= score <= 300:
        return "Orta risk"
    elif 301 <= score <= 400:
        return "Yüksek risk"
    else:
        return "En yüksek risk"

# Öneriler fonksiyonu
def make_recommendations(risk_level):
    if risk_level == "Düşük risk":
        print("Meme kanseri risk düzeyiniz düşük. Yıllık mamografi taramalarınızı yaptırmayı düşünebilirsiniz.")
    elif risk_level == "Orta risk":
        print("Meme kanseri risk düzeyiniz orta. Düzenli mamografi taramaları yanında doktorunuzla risk azaltma stratejileri üzerine görüşmelisiniz.")
    elif risk_level == "Yüksek risk":
        print("Meme kanseri risk düzeyiniz yüksek. Düzenli mamografi taramalarının yanı sıra, genetik danışmanlık almayı ve risk azaltıcı önlemler üzerine doktorunuzla konuşmayı düşünmelisiniz.")
    else:
        print("Meme kanseri risk düzeyiniz en yüksek. Acilen bir uzman hekime başvurarak daha ayrıntılı bir değerlendirme ve öneriler almalısınız.")

# Kullanıcıya soruları sorma ve sonucu gösterme
def main():
    print("Meme kanseri riski belirleme uygulamasına hoş geldiniz.")
    answers = {}
    
    # Yaş değerini direkt olarak al
    try:
        age = int(input("Lütfen yaşınızı girin: "))
    except ValueError:
        print("Geçerli bir yaş değeri giriniz.")
        return
    
    if age <= 0:
        print("Yaş değeri pozitif bir tam sayı olmalıdır.")
        return
    
    answers["Yaş"] = age
    print("**************************************")  # Yaş girdikten sonra ********** sembolü ile sorular bölünüyor
    
    for question, options in questions.items():
        if question != "Yaş":
            print(question.capitalize() + " için seçenekler:")
            for option, _ in options.items():
                print(option.capitalize())
            while True:
                answer = input("Lütfen bir seçenek girin: ").lower()  # Kullanıcının cevabını küçük harfe dönüştürme
                if answer in options:
                    answers[question] = options[answer]  # Cevap doğrudan puan değeri olarak kaydediliyor
                    print("****************************************")  # Her soru-cevap çiftinden sonra *** ekleniyor
                    break
                else:
                    print("Lütfen geçerli bir seçenek girin: " + ", ".join(options.keys()))  # Geçerli seçenekleri kullanıcıya gösterme
    
    # Puanları topla
    total_score = sum(answers.values())  # Doğrudan cevapların puanları toplanıyor

    # Risk düzeyini belirle
    risk_level = determine_risk_level(total_score)
    
    print("Meme kanseri risk düzeyiniz:", risk_level)
    
    # Önerileri göster
    make_recommendations(risk_level)

# Ana fonksiyonu çağırma
if __name__ == "__main__":
    main()
