### load libraries ###
{
library(tidyverse)
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(na.tools)
library(ggimage)
library(ggbeeswarm)
library(cfbfastR)
library(gt)
library(mgcv)
library(scales)
library(ggforce)
library(remotes)
library(ggtext)
library(bayesboot)
library(rvest)
}

## pull all cfb plays for current season ##

pbp <- cfbfastR::load_cfb_pbp()

## filter only rushing plays ##

pbp_rp <- pbp %>%
  filter(!is.na(EPA), rush == 1, penalty_text == "FALSE",!is.na(rusher_player_name), play_type %in% c("Rush", "Rushing Touchdown"))

## change clock configuration to match NFLfastR standards ##

pbp_rp <- pbp_rp %>%
  mutate(quarter_seconds_remaining = clock.minutes*60+clock.seconds,
         half_seconds_remaining = TimeSecsRem) %>%
  mutate(game_seconds_remaining = case_when(
    half == 1 ~ half_seconds_remaining+1800,
    TRUE ~ half_seconds_remaining
  ))

## create column of defensive yards allowed on rushing plays for each team ##

def_ypc <- pbp_rp %>%
  group_by(def_pos_team) %>%
  summarize(def_ypc = mean(yards_gained),
            count = n()) %>%
  filter(count >= 100) %>% ##filter to 100 plays or more
  select(-count)

## join in defensive yards per carry ##

rush_attempts <- pbp_rp %>%
  left_join(def_ypc, by = "def_pos_team")

## cap big gains and losses to 20yds and -5yds

rush_attempts3 <- rush_attempts %>%
  mutate(yards_rushed = case_when(yards_gained > 20 ~ 20L,
                                  yards_gained < -5 ~ -5L,
                                  TRUE ~ as.integer(yards_gained)),
         label = yards_rushed + 5L)

## feature selection for model ##

rush_attempts4 <- rush_attempts3 %>%
  select(yard_line, quarter_seconds_remaining, half_seconds_remaining,
         game_seconds_remaining, period, down, Goal_To_Go, distance,
         score_diff, ep_before, wp_before, def_ypc, label) %>%
  filter(!is.na(label), !is.na(down)) %>%
  mutate(Goal_To_Go = case_when(Goal_To_Go == "FALSE" ~ 0, TRUE ~ 1))

## run xgboost model

smp.size <- floor(.8*nrow(rush_attempts4))
set.seed(123)
ind <- sample(seq_len(nrow(rush_attempts4)),size=smp.size)
ind_train <- rush_attempts4[ind, ]
ind_test <- rush_attempts4[-ind, ]

full_train <- xgboost::xgb.DMatrix(as.matrix(ind_train %>% select(-label)), label = as.integer(ind_train$label))

nrounds <- 100
params <-
  list(
    booster = "gbtree",
    objective = "multi:softprob",
    eval_metric = c("mlogloss"),
    num_class = 26,
    eta = .012,
    gamma = 1,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth = 8,
    min_child_weight = 21
  )

ryoe_model <- xgboost::xgboost(params = params, data = full_train, nrounds = nrounds, verbose = 2)

imp <- xgb.importance(colnames(full_train), model = ryoe_model)
xgb.plot.importance(imp)

rushes_all <- rush_attempts4 %>% mutate(index = 1:n()) %>% select(-label)

## predict expected rushing yards using model on current season ##

exp_yds <- stats::predict(ryoe_model,as.matrix(rushes_all %>% select(
  yard_line,quarter_seconds_remaining,half_seconds_remaining,game_seconds_remaining,period,down,Goal_To_Go,distance,score_diff,ep_before,wp_before,def_ypc
))) %>%
  tibble::as_tibble() %>%
  dplyr::rename(prob = "value") %>%
  dplyr::bind_cols(purrr::map_dfr(seq_along(rushes_all$index), function(x) {
  tibble::tibble("xyds_rushed" = -5:20,
                 "down" = rushes_all$down[[x]],
                 "yard_line" = rushes_all$yard_line[[x]],
                 "quarter_seconds_remaining" = rushes_all$quarter_seconds_remaining[[x]],
                 "half_seconds_remaining" = rushes_all$half_seconds_remaining[[x]],
                 "game_seconds_remaining" = rushes_all$game_seconds_remaining[[x]],
                 "period" = rushes_all$period[[x]],
                 "Goal_To_Go" = rushes_all$Goal_To_Go[[x]],
                 "distance" = rushes_all$distance[[x]],
                 "score_diff" = rushes_all$score_diff[[x]],
                 "ep_before" = rushes_all$ep_before[[x]],
                 "wp_before" = rushes_all$wp_before[[x]],
                 "index" = rushes_all$index[[x]])
  })) %>%
  dplyr::group_by(.data$index) %>%
  dplyr::mutate(max_loss = dplyr::if_else(.data$yard_line < 95, -5L, as.integer(.data$yard_line - 99L)),
                max_gain = dplyr::if_else(.data$yard_line > 20, 20L, as.integer(.data$yard_line)),
                cum_prob = cumsum(.data$prob),
                prob = dplyr::case_when(.data$xyds_rushed == .data$max_loss ~ .data$prob,
                                        .data$xyds_rushed == .data$max_gain ~ 1 - dplyr::lag(.data$cum_prob, 1),
                                        TRUE ~ .data$prob),
                yard_line = .data$yard_line - .data$xyds_rushed) %>%
  dplyr::filter(.data$xyds_rushed >= .data$max_loss, .data$xyds_rushed <= .data$max_gain) %>%
  dplyr::select(-.data$cum_prob) %>%
  dplyr::summarise(x_rush_yards = sum(.data$prob * .data$xyds_rushed)) %>%
  ungroup()

rushes_all2 <- rushes_all %>% inner_join(exp_yds)

## inner join expected yards to create RYOE ##

pbp_all <- cbind(rushes_all2[,c("x_rush_yards")],pbp_rp) %>%
  mutate(ryoe = yards_gained - x_rush_yards)

## UNC only dataset ##

unc_pbp <- pbp_all %>%
  filter(pos_team == "North Carolina")

## evaluate averages for each RB ##

rusher_ryoe <- pbp_all %>%
  group_by(rusher_player_name,pos_team) %>%
  summarize(attempts=n(),yds = sum(yards_gained),tds = sum(touchdown),ypc = yds/attempts, epa = mean(EPA), ryoe = mean(ryoe))

unc_ryoe <- rusher_ryoe %>% filter(pos_team == "North Carolina")

### UNC RB BEESWARM

unc_pbp %>%
  group_by(rusher_player_name) %>% mutate(n=n()) %>% filter(n>5) %>% ungroup() %>%
  mutate(ryoe = case_when(ryoe > 20 ~ 20, TRUE ~ ryoe)) %>%
  ggplot(aes(rusher_player_name,ryoe,fill=ryoe))+
  geom_hline(yintercept=0,linetype="dashed",color="red",size=1,alpha=.8)+
  ggbeeswarm::geom_quasirandom(pch=21,size=4.5)+
  scale_y_continuous(breaks=seq(-10,20,5))+
  labs(x="",y="Rushing Yards Over Expected")+
  ggthemes::theme_fivethirtyeight()+
  theme(legend.position = "none",
        axis.text.x = element_text(size=15))


### TOP RB BEESWARM ###

pbp_all %>% filter(rusher_player_name != "TEAM") %>%
  group_by(rusher_player_name) %>% mutate(n=n()) %>% filter(n>100) %>% ungroup() %>%
  mutate(ryoe = case_when(ryoe > 25 ~ 25, TRUE ~ ryoe)) %>%
  ggplot(aes(rusher_player_name,ryoe,fill=ryoe))+
  geom_hline(yintercept=0,linetype="dashed",color="red",size=1,alpha=.8)+
  ggbeeswarm::geom_quasirandom(pch=21,size=4.5)+
  scale_y_continuous(breaks=seq(-10,25,5))+
  labs(x="",y="Rushing Yards Over Expected")+
  ggthemes::theme_fivethirtyeight()+
  theme(legend.position = "none")

mean(pbp_all$ryoe)
