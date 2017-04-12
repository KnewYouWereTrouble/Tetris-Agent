import java.util.*;
import java.io.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class PlayerSkeleton {

	/* RANDOM FOREST TRAINING PARAMETERS */
	private static final int NUM_TREES = 30;
	private static final int NUM_FEATURES = 8;
	private static final int NUM_TRAINING_FEATURES = 5;

	private static final int INITIAL_POPULATION = 10000;
	private static final int NUM_GA_TRAINING_SETS = NUM_TREES;
	private static final int GA_TRAINING_POPULATION = (int) (INITIAL_POPULATION * 0.1);

	private static final String WEIGHTS_FILE = "weights30.txt";

	private Weight[] optimisedWeights;
	private int[][] trainingFeatures;

	/* GAME PARAMETERS */
	public PlayerSkeleton() {
		// Uncomment this to perform training
		// performTraining();

		// Uncomment this to use pre-trained weights
		loadPreTrainedWeights();
	}

	private void loadPreTrainedWeights() {
		try {
			File file = new File(WEIGHTS_FILE);
			Scanner sc = new Scanner(file);

			trainingFeatures = new int[NUM_GA_TRAINING_SETS][NUM_TRAINING_FEATURES];
			optimisedWeights = new Weight[NUM_GA_TRAINING_SETS];

			for (int t = 0; t < NUM_TREES; t++) {
				double[] weights = new double[NUM_TRAINING_FEATURES];
				int[] features = new int[NUM_TRAINING_FEATURES];

				optimisedWeights[t] = new Weight(weights);
				optimisedWeights[t].score = sc.nextInt();

				for (int f = 0; f < NUM_TRAINING_FEATURES; f++) {
					weights[f] = sc.nextDouble();
					features[f] = sc.nextInt();
				}
				trainingFeatures[t] = features;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	private void performTraining() {
		Weight[] initialPopulation = generateRandomPopulation(INITIAL_POPULATION);

		trainingFeatures = baggingFeatureSubset();
		Weight[][] trainingPopulations = baggingPopulationSubset(initialPopulation);

		Weight[] optimisedWeights = new Weight[NUM_GA_TRAINING_SETS];

		ExecutorService executor = Executors.newFixedThreadPool(4);
		List<Future<Weight>> list = new ArrayList<Future<Weight>>();

		for (int t = 0; t < NUM_TREES; t++) {
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
		this.optimisedWeights = optimisedWeights;
	}

	private int[][] baggingFeatureSubset() {
		Random r = new Random();

		int[][] trainingFeatures = new int[NUM_GA_TRAINING_SETS][NUM_TRAINING_FEATURES];
		for (int treeIdx = 0; treeIdx < NUM_GA_TRAINING_SETS; treeIdx++) {
			trainingFeatures[treeIdx][0] = FeatureDelegate.COMPLETE_LINES_INDEX;
			for (int fIdx = 1; fIdx < NUM_TRAINING_FEATURES; fIdx++) {
				trainingFeatures[treeIdx][fIdx] = r.nextInt(NUM_FEATURES);
			}
		}
		return trainingFeatures;
	}

	private Weight[][] baggingPopulationSubset(Weight[] initialPopulation) {
		Random r = new Random();

		// 10 sets of 1000 weights
		Weight[][] trainingPopulations = new Weight[NUM_GA_TRAINING_SETS][GA_TRAINING_POPULATION];

		for (int treeIdx = 0; treeIdx < NUM_GA_TRAINING_SETS; treeIdx++) {
			for (int weightIdx = 0; weightIdx < GA_TRAINING_POPULATION; weightIdx++) {
				// random sampling with replacement - bagging
				trainingPopulations[treeIdx][weightIdx] = new Weight(initialPopulation[r.nextInt(initialPopulation.length)].weights);

				for (int f = 0; f < NUM_TRAINING_FEATURES; f++) {
					if (trainingFeatures[treeIdx][f] == FeatureDelegate.COMPLETE_LINES_INDEX) {
						trainingPopulations[treeIdx][weightIdx].weights[f] *= -1;
					}
				}
			}
		}
		return trainingPopulations;
	}

	private Weight[] generateRandomPopulation(int size) {
		Weight[] population = new Weight[size];
		for (int p = 0; p < size; p++) {
			double[] weights = new double[NUM_TRAINING_FEATURES];
			for (int w = 0; w < NUM_TRAINING_FEATURES; w++) {
				weights[w] = -Math.random();
			}
			population[p] = new Weight(weights);
		}
		return population;
	}

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
		for (int tree = 0; tree < NUM_TREES; tree++) {
			double score = 0;
			for (int f = 0; f < NUM_TRAINING_FEATURES; f++) {
				score += optimisedWeights[tree].weights[f] * fd.applyFeature(field, trainingFeatures[tree][f]);
			}
			double votePercentage = optimisedWeights[tree].score / (double) totalWeightsScore;
			totalScore += score * votePercentage;
		}
		return totalScore;
	}

	private int weightsScore(Weight[] weights) {
		int totalWeightsScore = 0;
		for (int w = 0; w < NUM_TREES; w++) {
			totalWeightsScore += weights[w].score;
		}
		return totalWeightsScore;
	}

	public static void main(String[] args) {
		State s = new State();
		new TFrame(s);
		PlayerSkeleton p = new PlayerSkeleton();

		while(!s.hasLost()) {
			s.makeMove(p.pickMove(s,s.legalMoves()));
			s.draw();
			s.drawNext(0,0);
			try {
				Thread.sleep(25);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println("You have completed " + s.getRowsCleared() + " rows.");
	}
}

class GeneticAlgorithm implements Callable<Weight> {

	private class EvaluateWeightTask implements Callable<Integer> {
		private static final int NUM_GAMES = 5;

		private Weight weight;
		private int[] features;

		public EvaluateWeightTask(Weight weight, int[] features) {
			this.weight = weight;
			this.features = features;
		}

		@Override
	    public Integer call() throws Exception {
			ExecutorService executor = Executors.newFixedThreadPool(NUM_GAMES);
			List<Future<Integer>> list = new ArrayList<Future<Integer>>();


			for (int g = 0; g < NUM_GAMES; g++) {
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

	private class PlayGameTask implements Callable<Integer> {
		private static final int MAX_MOVES_PER_GAME = 500;

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

			while(!s.hasLost() && numMoves < MAX_MOVES_PER_GAME) {
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

	private static final double CHILDREN_SIZE_PERCENTAGE = 0.3;
	private static final double TOURNAMENT_SIZE_PERCENTAGE = 0.1;
	private static final int NUM_GENERATIONS = 50;
	private static final double MUTAION_PROBABILITY = 0.05;
	private static final double MUTAION_MAGNITUDE = 0.2;

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
		Weight[] childrenPopulation = new Weight[(int) (population.length * CHILDREN_SIZE_PERCENTAGE)];

		for (int g = 0; g < NUM_GENERATIONS; g++) {
			for (int c = 0; c < childrenPopulation.length; c++) {
				Weight[] tournamentPopulation = baggingTournamentSubset(population);
				Arrays.sort(tournamentPopulation, new WeightComparator());

				//Perform crossover
				Weight crossover = weightedAvgCrossover(tournamentPopulation[tournamentPopulation.length - 1],
														tournamentPopulation[tournamentPopulation.length - 2]);
				//Perform mutation
				mutate(crossover);
				normalise(crossover);
				childrenPopulation[c] = crossover;
			}
			applyFitnessFunction(childrenPopulation, features);
			replaceLastN(population, childrenPopulation);
		}

		System.out.println(population[population.length - 1].score);

		for (int w = 0; w < features.length; w++) {
			System.out.print(population[population.length - 1].weights[w] + " " + features[w]);
			System.out.println();
		}

		return population[population.length - 1];
	}

	private void applyFitnessFunction(Weight[] population, int[] features) {
		ExecutorService executor = Executors.newFixedThreadPool(30);
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
		if(Math.random() < MUTAION_PROBABILITY) {
			Random r = new Random();
			int component = r.nextInt(crossover.weights.length);
			if (Math.random() < 0.5) {
				crossover.weights[component] *= 1 + MUTAION_MAGNITUDE;
			} else {
				crossover.weights[component] *= 1 - MUTAION_MAGNITUDE;
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

		Weight[] tournamentPopulation = new Weight[(int) (population.length * TOURNAMENT_SIZE_PERCENTAGE)];

		for (int idx = 0; idx < tournamentPopulation.length; idx++) {
			tournamentPopulation[idx] = population[r.nextInt(population.length)];
		}
		return tournamentPopulation;
	}

}


class FeatureDelegate {
	public static final int COMPLETE_LINES_INDEX = 3;

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

	private int getColumnHeight(int col, int[][] field) {
		for (int i = field.length - 1; i >= 0; i--) {
			if (field[i][col] != 0) return i;
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
