library(ggplot2)
library(Rmisc)

########################### NOMBRE DE CLUSTERS ####################################

## Importation des données ##
data_clusters2=read.table(
  file = "/home/fort/Bureau/nb_clusters_modifie.txt",
  header = T,
  sep = "\t",
  dec = "."
)
## Recuperation des donnees pour les rangs courbes ##
C=data_clusters2[data_clusters2$forme=='curved',]

## Convertion en facteur ##
C$DIP.DIR=as.factor(C$DIP.DIR)
C$densite=as.factor(C$densite)
C$mean_ratio = mean(C$ratio_rang)

## Calcul du SD ##
std_C = summarySE(C, measurevar="ratio_rang", groupvars=c("densite","DIP.DIR"))

ggplot(std_C, aes(x=DIP.DIR, y=ratio_rang, colour=DIP.DIR))+
  theme(text = element_text(size=15))+
  geom_errorbar(aes(ymin=ratio_rang-sd, ymax=ratio_rang+sd), width=.5) +
  geom_point()+
  facet_wrap(~densite)+
  xlab("DIP/DIR")+
  ylab("nombres de rangs détectés/nombre réel de rangs")+
  ggtitle("DBSCAN - Détection des rangs courbes")

## Calcul du SD ##
std_C_plantes = summarySE(C, measurevar="ratio_plantes", groupvars=c("densite","DIP.DIR"))

## Plot pour les plantes ##
ggplot(std_C_plantes, aes(x=DIP.DIR, y=ratio_plantes, colour=DIP.DIR))+
  theme(text = element_text(size=15))+
  geom_errorbar(aes(ymin=ratio_plantes-sd, ymax=ratio_plantes+sd), width=.5) +
  geom_point()+
  facet_wrap(~densite)+
  xlab("DIP/DIR")+
  ylab("nombres de plantes détectées/nombre réel de rangs")+
  ggtitle("Fuzzy clustering - détection des plantes")
    
