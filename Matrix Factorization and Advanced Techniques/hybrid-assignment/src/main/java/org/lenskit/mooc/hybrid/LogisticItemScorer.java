package org.lenskit.mooc.hybrid;

import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.lenskit.api.ItemScorer;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.bias.BiasModel;
import org.lenskit.bias.UserBiasModel;
import org.lenskit.data.ratings.RatingSummary;
import org.lenskit.results.Results;
import org.lenskit.util.collections.LongUtils;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Item scorer that does a logistic blend of a subsidiary item scorer and popularity.  It tries to predict
 * whether a user has rated a particular item.
 */
public class LogisticItemScorer extends AbstractItemScorer {
    private final LogisticModel logisticModel;
    private final BiasModel biasModel;
    private final RecommenderList recommenders;
    private final RatingSummary ratingSummary;

    @Inject
    public LogisticItemScorer(LogisticModel model, UserBiasModel bias, RecommenderList recs, RatingSummary rs) {
        logisticModel = model;
        biasModel = bias;
        recommenders = recs;
        ratingSummary = rs;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        List<ItemScorer> scorers = recommenders.getItemScorers();
        List<ResultMap> scorerResults = new ArrayList<>(scorers.size());
        for (ItemScorer scorer : scorers) {
            scorerResults.add(scorer.scoreWithDetails(user, items));
        }

        LongSet itemSet = LongUtils.asLongSet(items);
        Long2DoubleMap itemBiases = biasModel.getItemBiases(itemSet);
        double userBias = biasModel.getIntercept() + biasModel.getUserBias(user);

        List<Result> results = new ArrayList<>(items.size());
        for (long item : items) {
            double baselineScore = userBias + itemBiases.get(item);
            double[] vars = new double[2 + scorers.size()];
            vars[0] = baselineScore;

            int pop = ratingSummary.getItemRatingCount(item);
            vars[1] = pop > 0 ? Math.log10(pop) : 0;

            for (int i = 0; i < scorerResults.size(); i++) {
                Result score = scorerResults.get(i).get(item);
                vars[i + 2] = score == null ? 0 : score.getScore() - baselineScore;
            }

            results.add(Results.create(item, logisticModel.evaluate(1, vars)));
        }

        return Results.newResultMap(results);
    }
}
