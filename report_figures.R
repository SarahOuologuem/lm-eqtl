library(data.table)
library(ggplot2)
library(ggsignif)

dt <- fread("../mean_results_pearson.tsv")

ggplot(dt, aes(x = gene, color=method)) +
  geom_errorbar(aes(ymax = mean_r + sd_r, ymin = mean_r - sd_r),
                position = position_dodge(0.5), size=1.3, width=0.4) +
  geom_point(aes(y=mean_r), shape=18, size=5,
             position = position_dodge2(width = 0.5, preserve = "single")) +
  labs(x="Gene ID", y="Pearson r", title="Expression Prediction in 5-fold CV") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        text=element_text(size=18))


dt <- fread("../mean_results_spearman.tsv")

ggplot(dt, aes(x = gene, color=method)) +
  geom_errorbar(aes(ymax = mean_r + sd_r, ymin = mean_r - sd_r),
                position = position_dodge(0.5), size=1.3, width=0.4) +
  geom_point(aes(y=mean_r), shape=18, size=5,
             position = position_dodge2(width = 0.5, preserve = "single")) +
  labs(x="Gene ID", y="Spearman r", title="Expression Prediction in 5-fold CV") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        text=element_text(size=18))





dt <- fread("../mean_results_pearson.tsv")

# ns: p <= 1.0, *: 0.01 < p <= 0.05, **: 0.001 < p <= 0.01
ggplot(dt, aes(x=method, y=mean_r)) + geom_boxplot() +
  theme(text=element_text(size=18)) + ylim(0.0, 1.0) +
  labs(x="Method", y="Pearson r", title="Expression Prediction over 10 genes") +
  geom_signif(comparisons = list(c("LinRegr", "Ridge"), c("SNP", "Ridge"), c("SVR", "Ridge")), 
              annotations = c("**", "*", "ns"), y_position = c(0.75, 0.8, 0.9),
              textsize=6, size=0.8)


dt <- merge(dt[, mean(mean_r), by="method"], dt[, sd(mean_r), by="method"],
            by="method")
colnames(dt) <- c("method", "mean_r", "sd_r")

pairs <- list()

ggplot(dt, aes(x=method)) +
  geom_errorbar(aes(ymax = mean_r + sd_r, ymin = mean_r - sd_r),
                size=2.0, width=0.1) +
  geom_point(aes(y=mean_r), shape=18, size=8) +
  labs(x="Method", y="Pearson r", title="Expression Prediction over 10 genes") +
  theme(text=element_text(size=18)) +
  ylim(0.0, 1.0) +
  geom_signif(aes(y=mean_r), comparisons = list(c("LinRegr", "Ridge")), 
              annotations = c("***"))
