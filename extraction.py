import yake
import string
import re

def keyword_extraction(text, max_word):
    
    kw_extractor = yake.KeywordExtractor(lan="id", n =1, top=max_word)
    keywords = kw_extractor.extract_keywords(text)

    final_kw = []

    for i in keywords:
        kw = next(iter(i))
        kw = str(kw)
        kw = kw.lower()
        final_kw.append(kw)

    return final_kw


#text = "Program kerja praktik ini merupakan bagian dari Capstone Project yang merupakan "\
#"rangkaian Program Bangkit 2021 led by Google. Program Bangkit ini merupakan "\
#"bagian dari program pemerintah Merdeka Belajar Kampus Merdeka (MKBM) di bawah "\
#"Kementerian Pendidikan dan Kebudayaan (Mendikbud). Program Bangkit 2021 "\
#"bekerja sama dengan Google, Traveloka, Tokopedia, dan Gojek sebagai partner dalam "\
#"menyukseskan program ini sekaligus pendampingan dari Industri dalam proses "\
#"pengajaran."

#text2 = "ciri a veracity big data milik rentan sisi akurat validitas analisis dalam b value nilai olah referensi prayogo c https www wartaekonomi co id read akses mei contoh implementasi bidang pariwisata guna peta potensi kuat aspek sektor ekonomi kreatif contoh platform social media promosi destinasi wisata variety variasi karakteristik homogen struktur semistruktur kumpul sumber velocity acu cepat real time tingkat ubah cepat ubah variabel tipe volume hasil d e fitur oltp relational olap warehouse proses operasional informasional orientasi transaksi analisa fungsi hari butuh informasi jangka dukung putus guna kasir pramuniaga professional database knowledge workers manager analis view detail ter sumarisasi konsolidasi akses baca tulis unit kerja pendek sederhana query kompleks desain db orientasi aplikasi historis subjek fokus masuk rifzan vs datawarehouse beda fungsi robicomp com datwarehouse perbedaanny html"

#text = "artificial intelligence ai cerdas buat cabang ilmu bidang komputer isi dalam tekan pola pikir manusia kembang intellejen mesin simulasi cerdas milik manusia model mesin program pikir hal manusia contoh ai siri sistem asisten pribadi apple iphone ipad siri asisten pribadi suara wanita aktif suara ramah interaksi langsung rutinitas hari siri bantu kirim pesan buka aplikasi temu informasi tambah acara kalender arah buka panggil suara otomatis siri sistem mesin canggih oleh paham deepface facebook alah contoh ai teknologi deepface milik facebook ai fungsi kenal wajah orang postingan foto teknologi tanda foto manual ai laku ai identifikasi orang foto ai latih dasar data data dapat tanda orang foto hasil saran ai orang foto tuju ai latih milik data ai identifikasi foto tesla smartphone mobil milik teknologi cerdas buat ai temu mobil milik harga milik fitur canggih milik mampu prediksi kemudi inovasi teknologi sifat absoulut mobil milik cerdas buat tesla mobil canggih cocok milik impi mobil mobil film hollywood rekomendasi e commerse konsep terap ai jumpa salah satu rekomendasi produk e commerce belanja salah e commerce belanja produk produk rekomendasi untuk ai oleh data cari produk beli produk produk data proses konsep ai data mining ai rekomendasi produk produk pas dapat kerja cs ganti mesin masa zaman teknologi canggih kembang pesat berita baca kabar ai ganti peran asli cs ai atur konsumen tanggap dasar hadap lihat kilas mudah langgan ai jam henti langgan puas tunggu kerja cs keluh robot customer jabat tuju ganti cs robot pengagguran indoensia teknologi butuh kembang teknologi hidup damping manusia basis for comparison data tradisonal big data meaning data tradisional arsitektur teknologi ekstraksi data sumber data basis sql database rasional bantu hasil lapor analitik definisi simpan data lapor analitik hasil proses data tradisional big data teknologi diri volume velocity and variety data volume tentu data asal sumber beda velocity acu cepat proses data variety acu jenis data dukung jenis format data accepted data source terima homogen situs produk dbms sumber data heterogen situs jalan produk dbms beda terima jenis sumber transaksi bisnis media sosial informasi sensor data spesifik alat berat asal produk dbms accepted type of formats tangan data struktural data rasional terima jenis format data struktur data rasional data struktur dokumen teks email video audio data stock ticker transaksi uang time variant data kumpul gudang data identifikasi periode data historis lapor analitis big data milik dekat identifikasi data muat periode salah dekat atas data big proses file datar arsip tanggal dekat baik identifikasi data muat milik pilih data streaming simpan data historis distributed file system olah data data warehousing data tradisional makan terkadang butuh penuh selesai proses salah guna big data hdfs hadoop distributed file system definisi muat data sistem distribusi program peta kurang data struktur data struktur definisi tipe data atribut rekord tuple rekord milik field data entitas entitas kelompok relasi kelas entitas kelompok milik atribut deskripsi entitas skema milik sama format guna data struktur basis data rasional atur data ukur sistem crm customer relationship management erp enterprise resource planning cms content managemnt system data struktur model data data struktur data struktur teks file video email lapor report presentasi power point pesan suara voice mail memo citra data bentuk tipe apa ikut format atur alur contoh data tampil halaman web data testruktur atur beda data struktur struktur data struktur tampil baris kolom basis data rasional nomor tanggal string kira data usaha gartner butuh simpan mudah kelola lindung solusi data terstuktur tampil baris kolom basis data rasional gambar audio video file olah email spreadsgeets data usaha gartner butuh simpan sulit kelola lindung solusi lamasensor sensor unsur beda iot mesin canggih sensor definisi instrumen ubah iot jaring standar cenderung pasif sistem aktif integrasi dunia nyata contoh sensor sensor suhu temperature sensor suhu ukur energi panas sumber deteksi ubah suhu ubah ubah data mesin buat butuh suhu lingkung perangkat level tani suhu tanah faktor kunci tumbuh tanam sensor kelembaban humidity jenis sensor ukur uap air atmosfer udara gas sensor kelembaban temu sistem panas ventilasi dingin udara hvac domain industri rumah temu daerah rumah sakit stasiun meteorologi lapor prediksi cuaca sensor jarak proximity sensor jarak deteksi objek non kontak sensor jenis sensor pancar medan elektromagnetik sinar radiasi inframerah sensor jarak milik guna tarik ritel sensor proximity deteksi gerak langgan produk minat guna diberitahu diskon tawar khusus produk letak sensor sensor jarak parkir mal stadion bandara sedia parkir jalur rakit bahan kimia makan jenis industri akselerometer accelerometers accelerometer deteksi akselerasi objek laju ubah cepat objek hubung accelerometer deteksi ubah gravitasi kasing akselerometer pedometer pintar pantau armada gerak lindung anti curi siaga sistem benda diam pindah aktuator elemen mengkonversikan besar listrik analog besar cepat putar perangkat elektromagnetik hasil daya gerak hasil gerak robot contoh aktuator elektrik solenoid motor stepper motor dc brushless dc motors motor induksi motor sinkron tau kerja ganti mesin robot masa zaman kerja mesin robot pabrik pabrik robot mudah kerja cepat kerja contoh kerja kasir ganti mesin otomatis bayar basis ai iot manusia khawatir ganti kerja tenaga manusi tenaga robot imbang robot pikir rancang manusi hidup slaing untung kerja efektif kurang tenaga manusia bidang lingkung hidup terap pantau kualitas udara air kondisi atmosfer tanah cakup pantau satwa liar habitat tanggulang bakar hutan pantau bencana alam dll bidang tani kumpul data suhu data curah hujan kelembaban hama cepat angin muat tanah data data bantu optimalisasi tani ambil putus dasar informasi kumpul contoh tani pantau suhu kelembaban tanah repot pantau berkat internet of things sektor tani variety variasi big data acu data struktur struktur semistruktur kumpul sumber data kumpul spreadsheet database data hadir bentuk email pdf foto video audio posting sm variasi salah karakteristik big data velocity velocity dasar acu cepat data realtime prospek luas tingkat ubah hubung kumpul data masuk cepat berbedabeda sembur aktivitas volume volume salah ciri big data big data volume data hasil sumber platform media sosial proses bisnis mesin jaring interaksi manusia dll data simpan gudang data karakteristik big data varicity veracity big data milik rentan sisi akurat validitas dalam analisis big data hasil putus value big data milik nilai olah value nilai data tentu putus ambil proses data dunia bisnis nilai data nilai data sulit salah ambil langkah manfaat nilai data bisnis efisien machine learning artificial intelligence"
#result = keyword_extraction(text, 15)
#print (str(result))