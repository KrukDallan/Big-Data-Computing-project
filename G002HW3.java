import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class G002HW3
{
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// MAIN PROGRAM
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args)
    {

        if (args.length != 4)
        {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
                .map(x-> strToVector(x))
                .repartition(L)
                .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();
        // Check on parameters
        if(k+z+1 > N/L)
        {
            System.out.println("Error: check value of k and z");
            System.out.println("k+z+1 can't be greater than N/L (N = input size)");
            return;
        }
        // ----- Pring input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end-start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end-start) + " ms");

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// AUXILIARY METHODS
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method strToVector: input reading
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str)
    {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++)
        {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method euclidean: distance function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b)
    {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method MR_kCenterOutliers: MR algorithm for k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers (JavaRDD<Vector> points, int k, int z, int L)
    {
        long startRound1 = System.currentTimeMillis();
        //------------- ROUND 1 ---------------------------
        JavaRDD<Tuple2<Vector,Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition,k+z+1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector,Long>> c_w = new ArrayList<>();
            for(int i =0; i < centers.size(); ++i)
            {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i,entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1

        //------------- ROUND 2 ---------------------------
        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k+z)*L);
        elems.addAll(coreset.collect());
        long endRound1 = System.currentTimeMillis();

        long startRound2 = System.currentTimeMillis();
        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        ArrayList<Long> weights = new ArrayList<>(elems.size());
        ArrayList<Vector> T = new ArrayList<>(elems.size());
        for (Tuple2<Vector,Long> x : elems)
        {
            T.add(x._1());
            weights.add(x._2());
        }
        ArrayList<Vector> S = SeqWeightedOutliers(T, weights, k, z , 2);
        long endRound2 = System.currentTimeMillis();

        // ****** Measure and print times taken by Round 1 and Round 2, separately

        System.out.println("Time Round 1: " + (endRound1-startRound1) + " ms");
        System.out.println("Time Round 2: " + (endRound2-startRound2) + " ms");

        // ****** Return the final solution
        return S;
        //
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method kCenterFFT: Farthest-First Traversal
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT (ArrayList<Vector> points, int k)
    {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius =0;

        for (int iter=1; iter<k; iter++)
        {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++)
            {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i])
                {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist)
                {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeWeights: compute weights of coreset points
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers)
    {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for(int i = 0; i < points.size(); ++i)
        {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for(int j = 1; j < centers.size(); ++j)
            {
                if(euclidean(points.get(i),centers.get(j)) < tmp)
                {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method SeqWeightedOutliers: sequential k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector>SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, int alpha)
    {
        int n_iter=0;
        /*
         * Precompute distances for faster execution of Bz
         */
        double[][] distances = new double[P.size()][P.size()];
        for(int i=0; i<P.size(); i++)
        {
            for(int j=0; j<P.size(); j++)
            {
                distances[i][j] = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
            }
        }
        /*
         * Compute the initial value of r.
         * r is defined as the minimum distance between the first k+z+1 points in P divided by 2.
         */
        double raggio = Double.POSITIVE_INFINITY;
        for (int i=0; i<k+z+1; i++)
        {
            for (int j=i+1; j<k+z+1; j++)
            {
                double sqr_dist = Vectors.sqdist(P.get(i), P.get(j));
                if (raggio > sqr_dist)
                {
                    raggio = sqr_dist;
                }
            }
        }
        raggio = Math.sqrt(raggio)/2;
        System.out.println("Initial guess = " + raggio + " ");
        /*
         * Main loop
         */
        while(true)
        {
            /*
             * Initialize Wz (cumulative weight of Z) and Z (set of points yet to be covered)
             */
            long Wz = 0L;
            n_iter++;

            // Z <- P
            ArrayList<Integer> Z = new ArrayList<Integer>(P.size()); //just indices
            for(int i=0; i<P.size(); i++)
            {
                Z.add(i); //initialize indices
            }

            // S <- {}
            ArrayList<Vector> S = new ArrayList<Vector>();

            // Wz <- sum of weights in P
            for(long l : W)
            {
                Wz += l;
            }

            // beginning centers search
            while((S.size() < k) && (Wz > 0L))
            {
                long max = 0L;
                int idx_of_newcenter = -1;
                for(int i=0; i<P.size(); i++)
                {
                    // initialize Bz and keep track of its weight
                    ArrayList<Integer> Bz = new ArrayList<Integer>();
                    long ball_weight =0L;

                    for (int j : Z) //for each index in Z
                    {
                        if (distances[i][j]<=((1+2*alpha)*raggio)) //if the distance between point i and point of said index is <=(1+2alpha)r
                        {
                            Bz.add(j); //add the index to Bz
                        }
                    }
                    // adjust compute weight of the points in the ball
                    for(int index : Bz)
                    {
                        ball_weight += W.get(index);
                    }
                    if(ball_weight > max)
                    {
                        max = ball_weight;
                        idx_of_newcenter = i;
                    }
                }

                //add the new center
                S.add(P.get(idx_of_newcenter));

                // initialize Bz as list of indices
                ArrayList<Integer> Bz = new ArrayList<Integer>();
                for (int j : Z)
                {
                    if (distances[idx_of_newcenter][j]<=((3+4*alpha)*raggio))
                    {
                        Bz.add(j);
                    }
                }
                // remove Bz from Z
                ArrayList<Integer> to_remove = new ArrayList<>();
                for(int index : Bz)
                {
                    to_remove.add(index);
                    Wz -= W.get(index);
                }
                Z.removeAll(to_remove); //O(|Z|)
            }
            //loop end factor
            if(Wz <= z)
            {
                System.out.println("Final guess = " + raggio + " ");
                System.out.println("Number of guesses = " + n_iter + " ");
                return S;
            }
            //continue the loop
            else
            {
                raggio = 2*raggio;
            }
        }
    }


// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeObjective: computes objective function
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective (JavaRDD<Vector> points, ArrayList<Vector> centers, int z) // T
    {
        // For each partition l € [1..L], compute the distance of each point in l from all the centers and take the minimum
        // Then keep (return) the z largest distances for each partition
        // Collect all L*z returned distances (z for each partition)
        // Discard the z largest distances
        // Return the largest among the remaining distances

        List<Double> largest_distances = points.mapPartitions(
                //&&&&&& Return the z largest distances of points from their center &&&&&&
                //&&&&&& Perform calculation for each partition separately!         &&&&&&
                point_iter -> {
                    ArrayList<Double> dist_in_partition = new ArrayList<>();
                    while(point_iter.hasNext()) { //O(N/L) ~= O(sqrt(N))
                        Vector x = point_iter.next();

                        //calculate min of distances from each center
                        double min_dist = Double.POSITIVE_INFINITY;
                        for (Vector center : centers) { //O(k) ~= O(1)
                            if (Vectors.sqdist(center, x) < min_dist) //SQUARED distances!!
                                min_dist = Vectors.sqdist(center, x);
                        }
                        dist_in_partition.add(min_dist);
                    }
                    dist_in_partition.sort(Comparator.reverseOrder()); //sort the ArrayList O((N/L)log(N/L)); may be implem. in O(z*N/L) ~= O(N/L) if needed

                    ArrayList<Double> ret = new ArrayList<>(z);
                    for(int i=0; i<=z; i++)
                        ret.add(dist_in_partition.get(i));
                    return ret.iterator(); //return only the z largest distances
                })

                //&&&&&& collect all partitions into *one executor* &&&&&&
                //&&&&&& sort them in decreasing order              &&&&&&
                //&&&&&& then take only the largest z+1 distances   &&&&&&
                .takeOrdered(z+1, Comparator.reverseOrder());

        //&&&&&& return the last element i.e. the z+1-th largest distance &&&&&&
        return Math.sqrt(largest_distances.get(z)); //return the minimum among all distances
    }
}