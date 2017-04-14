import java.util.*;
import java.io.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

class Constants {

	/* RANDOM FOREST TRAINING PARAMETERS */
	public static final int NUM_TREES = 80;
	public static final int NUM_FEATURES = 9;
	public static final int NUM_TRAINING_FEATURES = 6;

	public static final int INITIAL_POPULATION = 10000;
	public static final int NUM_GA_TRAINING_SETS = NUM_TREES;
	public static final int GA_TRAINING_POPULATION = (int) (INITIAL_POPULATION * 0.1);

	public static final String WEIGHTS_FILE = "weights" + NUM_TREES + ".txt";

	/* GAME PARAMETERS */
	public static final int NUM_GAMES = 5;
	public static final int MAX_MOVES_PER_GAME = 500;

	/* GA TRAINING PARAMETERS */
	public static final double CHILDREN_SIZE_PERCENTAGE = 0.3;
	public static final double TOURNAMENT_SIZE_PERCENTAGE = 0.2;
	public static final int NUM_GENERATIONS = 50;
	public static final double MUTAION_PROBABILITY = 0.05;
	public static final double MUTAION_MAGNITUDE = 0.2;

	/* FEATURE DELEGATE */
	public static final int COMPLETE_LINES_INDEX = 3;

	/* PARALLELISATION */
	public static int NUM_TREE_THREADS = 4;
	public static int NUM_WEIGHTS_THREADS = 30;
	public static int NUM_GAME_THREADS = NUM_GAMES;
}

public class PlayerSkeleton {

	private Weight[] optimisedWeights;
	private int[][] trainingFeatures;

	public PlayerSkeleton() {
		// Uncomment this to perform training
		// performTraining();

		// Uncomment this to use pre-trained weights
		loadPreTrainedWeights();
	}

	/**
	 * Load pre-trained weights from the file
	 */
	private void loadPreTrainedWeights() {
		try {
			File file = new File(Constants.WEIGHTS_FILE);
			Scanner sc = new Scanner(file);

			trainingFeatures = new int[Constants.NUM_GA_TRAINING_SETS][Constants.NUM_TRAINING_FEATURES];
			optimisedWeights = new Weight[Constants.NUM_GA_TRAINING_SETS];

			for (int t = 0; t < Constants.NUM_TREES; t++) {
				double[] weights = new double[Constants.NUM_TRAINING_FEATURES];
				int[] features = new int[Constants.NUM_TRAINING_FEATURES];

				optimisedWeights[t] = new Weight(weights);
				optimisedWeights[t].score = sc.nextInt();

				for (int f = 0; f < Constants.NUM_TRAINING_FEATURES; f++) {
					weights[f] = sc.nextDouble();
					features[f] = sc.nextInt();
				}
				trainingFeatures[t] = features;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Train several heuristic functions for the random forest
	 */
	private void performTraining() {
		Weight[] initialPopulation = generateRandomPopulation(Constants.INITIAL_POPULATION);

		trainingFeatures = baggingFeatureSubset();
		Weight[][] trainingPopulations = baggingPopulationSubset(initialPopulation);

		// Final trained weights for the different heuristic functions
		optimisedWeights = new Weight[Constants.NUM_GA_TRAINING_SETS];

		ExecutorService executor = Executors.newFixedThreadPool(Constants.NUM_TREE_THREADS);
		List<Future<Weight>> list = new ArrayList<Future<Weight>>();

		for (int t = 0; t < Constants.NUM_TREES; t++) {
			Callable<Weight> callable = new GeneticAlgorithm(trainingPopulations[t], trainingFeatures[t]);
			Future<Weight> future = executor.submit(callable);
			list.add(future);
		}

		for(int i = 0; i < list.size(); i++){
			try {
				optimisedWeights[i] = list.get(i).get();
			} catch (InterruptedException | ExecutionException e) {
				e.printStackTrace();
			}
		}
		executor.shutdown();
	}

	/**
	 * Bootstrap Aggregate sample of the features
	 */
	private int[][] baggingFeatureSubset() {
		Random r = new Random();

		int[][] trainingFeatures = new int[Constants.NUM_GA_TRAINING_SETS][Constants.NUM_TRAINING_FEATURES];
		for (int treeIdx = 0; treeIdx < Constants.NUM_GA_TRAINING_SETS; treeIdx++) {
			trainingFeatures[treeIdx][0] = Constants.COMPLETE_LINES_INDEX;
			for (int fIdx = 1; fIdx < Constants.NUM_TRAINING_FEATURES; fIdx++) {
				trainingFeatures[treeIdx][fIdx] = r.nextInt(Constants.NUM_FEATURES);
			}
		}
		return trainingFeatures;
	}

	/**
	 * Bootstrap Aggregate sample of the population weights
	 */
	private Weight[][] baggingPopulationSubset(Weight[] initialPopulation) {
		Random r = new Random();

		// 10 sets of 1000 weights
		Weight[][] trainingPopulations = new Weight[Constants.NUM_GA_TRAINING_SETS][Constants.GA_TRAINING_POPULATION];

		for (int treeIdx = 0; treeIdx < Constants.NUM_GA_TRAINING_SETS; treeIdx++) {
			for (int weightIdx = 0; weightIdx < Constants.GA_TRAINING_POPULATION; weightIdx++) {
				// random sampling with replacement - bagging
				trainingPopulations[treeIdx][weightIdx] = new Weight(initialPopulation[r.nextInt(initialPopulation.length)].weights);

				for (int f = 0; f < Constants.NUM_TRAINING_FEATURES; f++) {
					if (trainingFeatures[treeIdx][f] == Constants.COMPLETE_LINES_INDEX) {
						trainingPopulations[treeIdx][weightIdx].weights[f] *= -1;
					}
				}
			}
		}
		return trainingPopulations;
	}

	/**
	 * Generate inital population randomly
	 */
	private Weight[] generateRandomPopulation(int size) {
		Weight[] population = new Weight[size];
		for (int p = 0; p < size; p++) {
			double[] weights = new double[Constants.NUM_TRAINING_FEATURES];
			for (int w = 0; w < Constants.NUM_TRAINING_FEATURES; w++) {
				weights[w] = -Math.random();
			}
			population[p] = new Weight(weights);
		}
		return population;
	}

	/**
	 * Game playing methods
	 */
	public int pickMove(State s, int[][] legalMoves) {
		int bestChoice = 0;
		double bestScore = Double.NEGATIVE_INFINITY;

		for (int choice = 0; choice < legalMoves.length; choice++) {
			int[][] field = new int[s.getField().length][];
			for (int i = 0; i < field.length; i++)
				field[i] = s.getField()[i].clone();

			int orient = legalMoves[choice][0];
			int slot = legalMoves[choice][1];

			// height if the first column makes contact
			int height = s.getTop()[slot] - State.getpBottom()[s.getNextPiece()][orient][0];
				// for each column beyond the first in the piece
			for (int c = 1; c < State.getpWidth()[s.getNextPiece()][orient]; c++) {
				height = Math.max(height, s.getTop()[slot + c] - State.getpBottom()[s.getNextPiece()][orient][c]);
			}

			// for each column in the piece - fill in the appropriate blocks
			for (int i = 0; i < State.getpWidth()[s.getNextPiece()][orient]; i++) {
				// from bottom to top of brick
				for (int h = height + State.getpBottom()[s.getNextPiece()][orient][i]; h < height
					 + State.getpTop()[s.getNextPiece()][orient][i]; h++) {
						 if (!(h >= field.length))
							field[h][i + slot] = s.getTurnNumber();
				}
			}

			double score = ensembleEvaluate(field);
			if (score > bestScore) {
				bestScore = score;
				bestChoice = choice;
			}
		}
		return bestChoice;
	}

	private double ensembleEvaluate(int[][] field) {
		FeatureDelegate fd = new FeatureDelegate();
		int totalWeightsScore = weightsScore(optimisedWeights);

		double totalScore = 0;
		for (int tree = 0; tree < Constants.NUM_TREES; tree++) {
			double score = 0;
			for (int f = 0; f < Constants.NUM_TRAINING_FEATURES; f++) {
				score += optimisedWeights[tree].weights[f] * fd.applyFeature(field, trainingFeatures[tree][f]);
			}
			double votePercentage = optimisedWeights[tree].score / (double) totalWeightsScore;
			totalScore += score * votePercentage;
		}
		return totalScore;
	}

	private int weightsScore(Weight[] weights) {
		int totalWeightsScore = 0;
		for (int w = 0; w < Constants.NUM_TREES; w++) {
			totalWeightsScore += weights[w].score;
		}
		return totalWeightsScore;
	}

	public static void main(String[] args) {
		// new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();

		int totalLines = 0;
		for (int i = 0; i < 100; i++) {
			State s = new State();
			while(!s.hasLost()) {
				s.makeMove(p.pickMove(s,s.legalMoves()));
				/*
				s.draw();
				s.drawNext(0,0);
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}*/
			}
			totalLines += s.getRowsCleared();
			System.out.println(s.getRowsCleared());
		}
		System.out.println(totalLines / 100);
	}
}

class GeneticAlgorithm implements Callable<Weight> {

	private Weight[] population;
	private int[] features;

	public GeneticAlgorithm(Weight[] population, int[] features) {
		this.population = population;
		this.features = features;
	}

	@Override
	public Weight call() throws Exception {
		return learn();
	}

	// This will return the optimised weights after GA
	private Weight learn() {
		applyFitnessFunction(population, features);
		Weight[] childrenPopulation = new Weight[(int) (population.length * Constants.CHILDREN_SIZE_PERCENTAGE)];

		for (int g = 0; g < Constants.NUM_GENERATIONS; g++) {
			for (int c = 0; c < childrenPopulation.length; c++) {
				Weight[] tournamentPopulation = baggingTournamentSubset(population);
				Arrays.sort(tournamentPopulation, new WeightComparator());

				// Perform crossover
				Weight crossover = weightedAvgCrossover(tournamentPopulation[tournamentPopulation.length - 1],
														tournamentPopulation[tournamentPopulation.length - 2]);
				// Perform mutation
				mutate(crossover);

				// Perform normalisation
				normalise(crossover);
				childrenPopulation[c] = crossover;
			}
			applyFitnessFunction(childrenPopulation, features);
			replaceLastN(population, childrenPopulation);
		}

		applyFitnessFunction(population, features);
		System.out.println(population[population.length - 1].score);

		for (int w = 0; w < features.length; w++) {
			System.out.print(population[population.length - 1].weights[w] + " " + features[w]);
			System.out.println();
		}

		return population[population.length - 1];
	}

	// Evalute multiple weights in parallel using ExecutorService
	private void applyFitnessFunction(Weight[] population, int[] features) {
		ExecutorService executor = Executors.newFixedThreadPool(Constants.NUM_WEIGHTS_THREADS);
		List<Future<Integer>> list = new ArrayList<Future<Integer>>();

		for (int w = 0; w < population.length; w++) {
			Callable<Integer> callable = new EvaluateWeightTask(population[w], features);
			Future<Integer> future = executor.submit(callable);
			list.add(future);
		}

		for(int w = 0; w < list.size(); w++){
			try {
				population[w].score = list.get(w).get();
			} catch (InterruptedException | ExecutionException e) {
				e.printStackTrace();
			}
		}
		executor.shutdown();
	}

	private Weight weightedAvgCrossover(Weight fittest, Weight secondFittest) {
		double[] crossover = new double[fittest.weights.length];

		for (int w = 0; w < fittest.weights.length; w++) {
			crossover[w] = fittest.weights[w] * fittest.score
						 + secondFittest.weights[w] * secondFittest.score;
		}
		return new Weight(crossover);
	}

	private void mutate(Weight crossover) {
		if(Math.random() < Constants.MUTAION_PROBABILITY) {
			Random r = new Random();
			int component = r.nextInt(crossover.weights.length);
			if (Math.random() < 0.5) {
				crossover.weights[component] *= 1 + Constants.MUTAION_MAGNITUDE;
			} else {
				crossover.weights[component] *= 1 - Constants.MUTAION_MAGNITUDE;
			}
		}
	}

	private void normalise(Weight crossover) {
		double total = 0;
		for (int i = 0; i < crossover.weights.length; i++) {
			total += Math.pow(crossover.weights[i], 2);
		}
		total = Math.sqrt(total);

		for (int j = 0; j < crossover.weights.length; j++) {
			crossover.weights[j] /= total;
		}
	}

	private void replaceLastN(Weight[] population, Weight[] childrenPopulation) {
		Arrays.sort(population, new WeightComparator());
		for (int r = 0; r < childrenPopulation.length; r++) {
			population[r + population.length - childrenPopulation.length] = childrenPopulation[r];
		}
	}

	private Weight[] baggingTournamentSubset(Weight[] population) {
		Random r = new Random();

		Weight[] tournamentPopulation = new Weight[(int) (population.length * Constants.TOURNAMENT_SIZE_PERCENTAGE)];

		for (int idx = 0; idx < tournamentPopulation.length; idx++) {
			tournamentPopulation[idx] = population[r.nextInt(population.length)];
		}
		return tournamentPopulation;
	}


	/**
	 * Callable task of evalutating a particular weight vector.
	 * Made up of the callable task of actually applying the fitness function by playing the game
	 */
	private class EvaluateWeightTask implements Callable<Integer> {

		private Weight weight;
		private int[] features;

		public EvaluateWeightTask(Weight weight, int[] features) {
			this.weight = weight;
			this.features = features;
		}

		@Override
		public Integer call() throws Exception {
			ExecutorService executor = Executors.newFixedThreadPool(Constants.NUM_GAME_THREADS);
			List<Future<Integer>> list = new ArrayList<Future<Integer>>();


			for (int g = 0; g < Constants.NUM_GAMES; g++) {
				Callable<Integer> callable = new PlayGameTask(weight, features);
				Future<Integer> future = executor.submit(callable);
				list.add(future);
			}

			int fitnessScore = 0;
			for(Future<Integer> fut : list){
				try {
					fitnessScore += fut.get();
				} catch (InterruptedException | ExecutionException e) {
					e.printStackTrace();
				}
			}
			executor.shutdown();
			return fitnessScore;
		}
	}

	/**
	 * Callable task of applying the fitness function on a particular weight by playing games on it
	 */
	private class PlayGameTask implements Callable<Integer> {

		private Weight weight;
		private int[] features;
		private FeatureDelegate fd;

		public PlayGameTask(Weight weight, int[] features) {
			this.weight = weight;
			this.features = features;
			this.fd = new FeatureDelegate();
		}

		@Override
		public Integer call() throws Exception {
			return fitnessFunction(weight);
		}

		private int fitnessFunction(Weight weight) {
			State s = new State();
			int numMoves = 0;

			while(!s.hasLost() && numMoves < Constants.MAX_MOVES_PER_GAME) {
				s.makeMove(this.pickMove(s, s.legalMoves(), weight));
				numMoves++;
			}
			return s.getRowsCleared();
		}

		private int pickMove(State s, int[][] legalMoves, Weight weight) {
			int bestChoice = 0;
			double bestScore = Double.NEGATIVE_INFINITY;

			for (int choice = 0; choice < legalMoves.length; choice++) {
				int[][] field = new int[s.getField().length][];
				for (int i = 0; i < field.length; i++)
					field[i] = s.getField()[i].clone();

				int orient = legalMoves[choice][0];
				int slot = legalMoves[choice][1];

				// height if the first column makes contact
				int height = s.getTop()[slot] - State.getpBottom()[s.getNextPiece()][orient][0];
					// for each column beyond the first in the piece
				for (int c = 1; c < State.getpWidth()[s.getNextPiece()][orient]; c++) {
					height = Math.max(height, s.getTop()[slot + c] - State.getpBottom()[s.getNextPiece()][orient][c]);
				}

				// for each column in the piece - fill in the appropriate blocks
				for (int i = 0; i < State.getpWidth()[s.getNextPiece()][orient]; i++) {
					// from bottom to top of brick
					for (int h = height + State.getpBottom()[s.getNextPiece()][orient][i]; h < height
						 + State.getpTop()[s.getNextPiece()][orient][i]; h++) {
							 if (!(h >= field.length))
								field[h][i + slot] = s.getTurnNumber();
					}
				}

				double score = evaluate(weight.weights, features, field);
				if (score > bestScore) {
					bestScore = score;
					bestChoice = choice;
				}
			}
			return bestChoice;
		}

		private double evaluate(double[] weights, int[] features, int[][] field) {
			double score = 0;
			for (int fIdx = 0; fIdx < features.length; fIdx++) {
				score += fd.applyFeature(field, features[fIdx]) * weights[fIdx];
			}
			return score;
		 }
	}

}


class FeatureDelegate {

	public int applyFeature(int[][] field, int feature) {
		switch (feature) {
			case 0:
			return aggregateHeight(field);

			case 1:
			return avgHeight(field);

			case 2:
			return maxMinHeightDiff(field);

			case 3:
			return numCompleteLines(field);

			case 4:
			return numHoles(field);

			case 5:
			return totalBlocks(field);

			case 6:
			return heightVariance(field);

			case 7:
			return bumpiness(field);

			case 8:
			return blockades(field);
		}
		return 0;
	}

	private int aggregateHeight(int[][] field) {
		int aggregateHeight = 0;
		for (int c = 0; c < field[0].length; c++) {
			aggregateHeight += getColumnHeight(c, field);
		}
		return aggregateHeight;
	}

	private int avgHeight(int[][] field) {
		return aggregateHeight(field) / State.COLS;
	}

	private int maxMinHeightDiff(int[][] field) {
		int max = getColumnHeight(0, field);
		int min = getColumnHeight(0, field);

		for (int c = 0; c < State.COLS; c++) {
			int height = getColumnHeight(c, field);
			if (height > max) max = height;
			if (height < min) min = height;
		}
		return Math.abs(max - min);
	}

	private int numCompleteLines(int[][] field) {
		int numCompleteLines = 0;

		for (int r = 0; r < field.length; r++) {
			boolean complete = true;
			for (int c = 0; c < field[0].length; c++) {
				if (field[r][c] == 0) {
					complete = false;
					break;
				}
			}
			if (complete) numCompleteLines++;
		}
		return numCompleteLines;
	}

	private int numHoles(int[][] field) {
		int numHoles = 0;

		for (int c = 0; c < field[0].length; c++) {
			int columnHeight = getColumnHeight(c, field);
			for (int r = 0; r < columnHeight; r++) {
				if (field[r][c] == 0) numHoles++;
			}
		}
		return numHoles;
	}

	private int totalBlocks(int[][] field) {
		int count = 0;
		for (int r = 0; r < State.ROWS; r++) {
			for (int c = 0; c < State.COLS; c++) {
				if (field[r][c] != 0) count++;
			}
		}
		return count;
	}

	private int heightVariance(int[][] field) {
		int avgHeight = avgHeight(field);
		int sum = 0;
		for (int c = 0; c < field[0].length; c++) {
			int columnHeight = getColumnHeight(c, field);
			sum += (columnHeight - avgHeight) * (columnHeight - avgHeight);
		}
		return sum / State.COLS;
	}

	private int bumpiness(int[][] field) {
		int bumpiness = 0;

		for (int c = 0; c < field[0].length - 1; c++) {
			int columnHeightCurrent = getColumnHeight(c, field);
			int columnHeightNext = getColumnHeight(c + 1, field);
			bumpiness += Math.abs(columnHeightCurrent - columnHeightNext);
		}
		return bumpiness;
	}

	private int blockades(int[][] field) {
		int bloackades = 0;

		for (int c = 0; c < field[0].length - 1; c++) {
			int columnHeightCurrent = getColumnHeight(c, field);
			for (int b = 0; b < columnHeightCurrent; b++) {
				if (field[b][c] == 0) {
					bloackades += columnHeightCurrent - b;
					break;
				}
			}
		}
		return bloackades;
	}

	private int getColumnHeight(int col, int[][] field) {
		for (int i = field.length - 1; i >= 0; i--) {
			if (field[i][col] != 0) return i + 1;
		}
		return 0;
	}

}

class Weight {
	public double[] weights;
	public int score;

	public Weight(double[] weights) {
		this.weights = weights;
		score = 0;
	}
}

class WeightComparator implements Comparator<Weight> {
	public int compare(Weight w1, Weight w2) {
		return w1.score - w2.score;
	}

	public boolean equals(Object obj) {
		return this == obj;
	}
}
