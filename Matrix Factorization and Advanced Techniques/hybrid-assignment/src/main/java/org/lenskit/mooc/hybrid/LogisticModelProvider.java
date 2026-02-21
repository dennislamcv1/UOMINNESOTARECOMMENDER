package org.lenskit.mooc.hybrid;

import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.Rating;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.inject.Transient;
import org.lenskit.util.ProgressLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Trainer that builds logistic models.
 */
public class LogisticModelProvider implements Provider<LogisticModel> {
    private static final Logger logger = LoggerFactory.getLogger(LogisticModelProvider.class);
    private static final double LEARNING_RATE = 0.00005;
    private static final int ITERATION_COUNT = 100;

    private final LogisticTrainingSplit dataSplit;
    private final BiasModel baseline;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;
    private final int parameterCount;
    private final Random random;

    @Inject
    public LogisticModelProvider(@Transient LogisticTrainingSplit split,
                                 @Transient UserBiasModel bias,
                                 @Transient RecommenderList recs,
                                 @Transient RatingSummary rs,
                                 @Transient Random rng) {
        dataSplit = split;
        baseline = bias;
        recommenders = recs;
        ratingSummary = rs;
        parameterCount = 1 + recommenders.getRecommenderCount() + 1;
        random = rng;
    }

    @Override
    public LogisticModel get() {
        List<ItemScorer> scorers = recommenders.getItemScorers();
        double intercept = 0;
        double[] params = new double[parameterCount];
        LogisticModel current = LogisticModel.create(intercept, params);
        List<Rating> tuneRatings = new ArrayList<>(dataSplit.getTuneRatings());
        Map<Long, double[]> varsByRating = new HashMap<>(tuneRatings.size());

        logger.info("precomputing features for {} logistic training examples", tuneRatings.size());
        ProgressLogger cacheProgress = ProgressLogger.create(logger)
                                                     .setLabel("logistic feature cache")
                                                     .setCount(tuneRatings.size())
                                                     .start();
        for (Rating rating : tuneRatings) {
            long user = rating.getUserId();
            long item = rating.getItemId();

            double baselineScore = baseline.getIntercept() + baseline.getUserBias(user) + baseline.getItemBias(item);
            double[] vars = new double[parameterCount];
            vars[0] = baselineScore;

            int pop = ratingSummary.getItemRatingCount(item);
            vars[1] = pop > 0 ? Math.log10(pop) : 0;

            for (int i = 0; i < scorers.size(); i++) {
                Result score = scorers.get(i).score(user, item);
                vars[i + 2] = score == null ? 0 : score.getScore() - baselineScore;
            }

            varsByRating.put(rating.getId(), vars);
            cacheProgress.advance();
        }
        cacheProgress.finish();

        ProgressLogger trainProgress = ProgressLogger.create(logger)
                                                     .setLabel("logistic training")
                                                     .setCount(ITERATION_COUNT)
                                                     .setPeriod(5)
                                                     .start();
        for (int iter = 0; iter < ITERATION_COUNT; iter++) {
            Collections.shuffle(tuneRatings, random);

            for (Rating rating : tuneRatings) {
                double y = rating.getValue();
                double[] vars = varsByRating.get(rating.getId());
                double deltaBase = LEARNING_RATE * y * current.evaluate(-y, vars);

                intercept += deltaBase;
                for (int j = 0; j < params.length; j++) {
                    params[j] += deltaBase * vars[j];
                }

                current = LogisticModel.create(intercept, params);
            }

            trainProgress.advance();
        }
        trainProgress.finish();
        logger.info("trained logistic model: {}", current);

        return current;
    }

}
