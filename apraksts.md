# Idejas apraksts (max 500 vārdi)
Mūsu projekts ir AIZO (Atklātās Inženierzinātņu Olimpiādes) papīra formāta darbu automātiska labošana. Šī programma ir domāta testa atbilžu labošanai, ar papildus lauku labošanu, kā piemēram atbildes, kur skolēns ieraksta kādu skaitli.
Lai izmantotu programmu, autorizēts lietotājs iesniegtu attēlus ar darbiem mājaslapā, attēli nonāk līdz serverim, kur no tiem tiek iegūtas atzīmētās atbildes, kas tiek salīdzinātas ar pareizajām atbildēm. Attiecīgā skolēna atbildes un punkti tiek reģistrēti datubāzē, kur vēlāk no tās tos var iegūt, lai viegli varētu izveidot skolēna koda un iegūto punktu pa priekšmetiem tabulu, kas tiks publicēta mājaslapā. Pareizās atbildes tiek iegūtas ieskenējot lapu, kurā tās ir atzīmētas.
Atbilžu lapu formāts būtu tests, kurā pareizās atbildes atsevišķā atbilžu lapā atzīmē ar aplīti. Lauki, kuros skolēns ieraksta atbildi būtu atsevišķas kastītes.<br>
Šis projekts tiek izstrādāts, jo pašreizējais darbu labošanas veids ir laikietilpīgs un skolotājiem netīkams. Izstrādājot šo darbu mazāk laika tiks patērēts labojot darbus, kā rezultātā skolotāji varēs vairāk laiku veltīt uzdevumu izveidei, vai arī citu skolas darbu labošanai. Arī olimpiādes pildītājiem šis ļaus iegūt papildus statistiku, aplūkojot iegūtos punktus pa priekšmetiem, neuzliekot papildus slogu skolotājiem, jo šī statistika tiks iegūta automātiski.
# Problēmas analīze (min 100, max 300 vārdi) - mērķauditorija/lietotāji; kāpēc vērts risinājumu izstrādāt
Mūsu mērķauditorija ir Inženierzinātņu vidusskolas skolotāji un pastarpināti arī AIZO pildītāji. No vairākiem skolotājiem ir dzirdēts, cik kaitinoši un laikietiplīgi ir izlabot visus AIZO darbus. Tieši tāpēc arī tiek izstrādāts šis risinājums. Mūsu produkts atvieglos šo AIZO darbu labošanu, kā arī ļaus skolēniem apskatīties papildus statistiku par saviem darbiem, piemēram, atsevišķi punktus pa priekšmetiem, kā matemātika, fizika un ķīmija. 
Ar mūsu risinājumu darbus varēs ar vairākām ierīcēm vienlaicīgi ieskenēt, un cilvēki, kas skenēs pat varēs nebūt skolotāji, bet gan skolēni. Papildus, varēs arī izsekot līdzi, vai kāds darbs nav pazudis, kā piemēram, bija dzirdēts stāsts, ka jau kādu laiku pēc olimpiādes starp formulu lapām tika atrastas trīs atbilžu lapas.
# ER modelis relāciju DB vismaz 3 tabulas
<img width="1134" height="347" alt="image" src="https://github.com/user-attachments/assets/241591de-e7e3-4cde-898b-bc75dcf27f3a" />


# Izmantotās tehnoloģijas
- OpenCV - atbilžu atrašana lapā
- Tensorflow - MI modeļu trenēšana rokraksta/atbilžu atpazīšanai
- Mobilās ierīces - darbu nofotogrāfēšana, attēli pēc tam tiek iesūtīti mājaslapā
- Serveris - bildes un datu apstrādāšana, datu uzglabāšana 
# Īss plāns ~10 darba nedēļām (tabula: darba apraksts, datums, autors)
| Datums      | Darba apraksts |
| ----------- | :----------- |
| 02.02-08.02 | Rihards: Iepazīšanās ar bibliotēkām un pieejām MI pielietošanā rokrasta atpazīšanai <br>Mathers: Izveidot Flask serveri, kur var iesūtīt pareizās atbildes un paveiktos darbus |
| 09.02-15.02 | Rihards: Datu ievākšana modeļu trenēšanai <br>Mathers:Iztestēt Flask servera spējas(piemēram, vai ir iespējams augšuplādēt 500mb ar bildēm vienlaicigī), veikt nepieciešamos uzlabojumus |
| 16.02-22.02 | Rihards: Rokrasta ciparu atpazīšanas modeļa trenēšana <br>Mathers:Izveidot JSON labotāju, kas automatiski labo darbus |
| 23.02-01.03 | Rihards: Rokrasta ciparu atpazīšanas modeļa testēšana un uzlabošana <br>Mathers: Testēt apjomu un robežgadījumus JSON labotājām, veikt nepieciešamos uzlabojumus|
| 02.03-08.03 | Rihards: OpenCV izmantošana, lai lapā atrast pareizo atbilžu "kastītes" <br>Mathers:SQL datubāzes savienošana datu uzkrāšanai|
| 09.03-15.03 | Rihards: Testa atbilžu atpazīšana lapā (iekrāsoti aplīši) <br>Mathers: Flask servera papildināšana testa rezultātu publicēšanai |
| 16.03-22.03 | Rihards: Pārliecības robežu testēšana un modeļu uzlabošana <br>Mathers:Pirmie pilnas programmas testi(augšuplādēta bilde->ģenerēts JSON->izlabots JSON->saglabāti un publicēti rezultāti|
| 23.03-29.03 | Rihards: Implementēt metodes, ko veikt, ja atbildes nav skaidri salasāmas/saprotamas; modeļu savienošana ar servera loģiku <br>Mathers:Apdomāt un veikt potenciālus uzlabojumus visas programmas darbībai. Finalizēt visu frontendu|
| 30.03-05.03 | Rihards un Mathers: Testu veikšana, gala uzlabojumu veikšana |
| 06.02-12.03 | Rihards un Mathers: Gala produkta sagatavošana (docker), dokumentācijas pabeigšana|
# Piegādes formāts
Lietotājiem tā būs mājaslapa, kur augšuplādē sākotnēji darbu ar pareizajām atbildēm un tad veiktos darbus. Augšuplādetā informācija tiks procesētu lai automātiski izlabotu visus darbus un publicētu rezultātus


# Patstāvīgais darbs, funkcionālās un nefunkcionālās prasības
### Rihards Bukovskis
##### Funkcionālās prasības:
- Pareizās atbilžu lapas var iesniegt tikai autorizēti lietotāji
- Ja MI labotājs nespēj nolasīt atbildi, tam ir jāatgriež lietotājam kļūda un jāliek lietotājam pašam ievadīt atbildi
- Autorizētiem lietotājiem ir jāspēj pārskatīt lietotāju iegūtos punktus
##### Nefunkcionālās prasības
- Pat ja lietotājs vienlaicīgi iesniedz vairākus attēlus, backend labošanai būtu jāspēj visi apstrādāt
- Katrs darbs ir jāizlabo 2 sekunžu laikā backendam
- Lietotāju UI ir jābūt intuitīvai, lai procesu padarītu ērtāku un samazinātu kļūdu iespējamību


###Edvards Mathers
##### Funckionālās prasības:
-Sistēma ļauj augšuplādēt pareizās atbildes un tās atpazīst no pārejiem pildītajiem darbiem
-Sistēma ļauj vienlaicīgi augšuplādēt vairākas bildes 
-Sistēmai jāspēj atpazīt, kura atbilde ir atzīmēta testā un skaitliski ievadīti skaitļi
##### Nefunkcionālās prasības:
-Atbilžu detektēšanai un labošanai jābut uzticamai, lai manuāla pārbaude ir nepieciešama tikai retos gadījumos.
-Serverim jāspēj darboties ar vairākiem lietotājiem vienlaicīgi un ~300 augsuplādētiem attēliem.
-Sistēmai jābūt drošai, lai to nav iespējams neparedzēti ietekmēt vai modificēt
