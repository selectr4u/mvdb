#include "vdb.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* (internal) calculate dot product */
vdb_embedding_t calculate_dot_product(vdb_embedding_t vec1[],
                                      vdb_embedding_t vec2[], unsigned int n) {
  /* for future may want to consider SIMD (single instruction multiple data for
   * efficiency) */
  vdb_embedding_t sum = 0;
  int i;
  for (i = 0; i < n; i++) {
    sum += vec1[i] * vec2[i];
  }

  return sum;
}

/* (internal) calculate magnitude */
vdb_embedding_t calculate_magnitude(vdb_embedding_t vec1[], unsigned int n) {
  vdb_embedding_t dp, mag;

  dp = calculate_dot_product(vec1, vec1, n);
  mag = sqrt(dp);

  return mag;
}

/* (internal) set error message */
void set_error_msg(vdb_instance *vdb, char *string, ...) {
  va_list args;
  va_start(args, string);
  vsnprintf(vdb->err, sizeof(string), string, args);
  va_end(args);
}

/* creates a new vec database instance */
vdb_instance *vdb_new(unsigned int vec_size) {
  vdb_instance *vdb = malloc(sizeof(vdb_instance));
  if (vdb == NULL) {
    return NULL;
  }

  vdb->capacity = VDB_INITIAL_ARRAY_CAPACITY;
  vdb->size = 0;
  vdb->id_counter = 1; /* no vec will have id 0 */
  vdb->vector_size = vec_size;

  /* create vec_array with vdb_vector capacity of 5 (the initial array capacity)
   */
  vdb_vector **vec_array =
      malloc(sizeof(vdb_vector *) * VDB_INITIAL_ARRAY_CAPACITY);
  if (vec_array == NULL) {
    free(vdb);
    return NULL;
  }

  vdb->vector_array = vec_array;

  return vdb;
}

/* handles creating a new vector (allocation) */
vdb_vector *vdb_new_vec(vdb_embedding_t vec_embeddings[], unsigned int size) {
  vdb_vector *vec = malloc(sizeof(vdb_vector));
  if (vec == NULL) {
    return NULL;
  }

  vec->size = size;
  vdb_embedding_t *v_embedding = malloc(sizeof(vdb_embedding_t) * size);
  if (v_embedding == NULL) {
    return NULL;
  }

  /* copy all embeddings to the allocated vector */
  memcpy(v_embedding, vec_embeddings,
         size * sizeof(vdb_embedding_t)); /* size is amount of vectors but not
                                   amount of bytes so we * by size of
                                   vdb_embedding_t to get total bytes */

  vec->vec_embeddings = v_embedding;

  return vec;
}

/* inserts a vector into the vec database */
vdb_vec_id vdb_insert(vdb_instance *vdb, vdb_vector *vec) {
  /* verify size match */
  if (vdb->vector_size != vec->size) {
    set_error_msg(vdb,
                  "vdb: vec size msimatch between db and vec: %d versus %d",
                  vdb->vector_size, vec->size);
    return VDB_INVALID_ID;
  }

  /* check if we need to resize*/
  if ((vdb->capacity - 1) >= vdb->size) {
    /* resize */
    vdb->capacity = vdb->capacity * VDB_ARRAY_RESIZE_MULTIPLIER;
    void *new_vec_array = realloc(vdb->vector_array, vdb->capacity);
    if (new_vec_array == NULL) {
      set_error_msg(vdb, "vdb: unable to allocate new memory!");
      return VDB_INVALID_ID;
    }

    vdb->vector_array = new_vec_array;
  }

  /* insert */
  vec->id = vdb->id_counter;
  vdb->id_counter++;
  vdb->size++;

  vdb->vector_array[vdb->size] = vec;

  return vec->id;
}

/* returns a pointer intended not to be modified or freed (read-only) */
const vdb_vector *vdb_get(vdb_instance *vdb, vdb_vec_id id) {
  vdb_vector *vec = NULL;

  /* first try direct access with id-1 (since the way id counter works allows
   * this) */
  vec = vdb->vector_array[id - 1];

  if (vec != NULL && vec->id == id) {
    return vec;
  }

  /* otherwise do the loop */
  vec = NULL;

  /* because of the way vec_id works, we can loop up until vec_id */
  int i;

  for (i = 0; i < id; i++) {
    vec = vdb->vector_array[i];
    if (vec != NULL) {
      if (vec->id == id) {
        break;
      }
    }
  }

  return vec;
}

/* removes from vec database and returns pointer */
vdb_vector *vdb_remove(vdb_instance *vdb, vdb_vec_id id) {
  vdb_vector *vec = NULL;

  /* first try direct access with id-1 (since the way id counter works allows
   * this) */
  vec = vdb->vector_array[id - 1];

  if (vec != NULL && vec->id == id) {
    vdb->vector_array[id - 1] = NULL; /* signify emptiness */
    vdb->size--;
    return vec;
  }

  /* otherwise do the loop */
  vec = NULL;

  /* because of the way vec_id works, we can loop up until vec_id */
  int i;

  for (i = 0; i < id; i++) {
    vec = vdb->vector_array[i];
    if (vec != NULL) {
      if (vec->id == id) {
        /* set that item pos to NULL to signify emptiness */
        vdb->vector_array[i] = NULL;
        break;
      }
    }
  }

  vdb->size--;
  return vec;
}

/* comparison function for vdb_top_n */
int _vdb_compare(const void *arg1, const void *arg2) {
  /* both arg1 and arg2 are vdb_comparison_result */
  vdb_comparison_result vec_c1 = *(vdb_comparison_result *)arg1;
  vdb_comparison_result vec_c2 = *(vdb_comparison_result *)arg2;

  if (vec_c1.similarity < vec_c2.similarity) {
    return 1; /* to get in BIGGEST->SMALLEST order */
  }

  if (vec_c1.similarity > vec_c2.similarity) {
    return -1; /* to get in BIGGEST->SMALLEST order */
  }

  return 0;
}

/* gets top n vecs using specified comparison function */
vdb_top_n_results vdb_top_n(vdb_instance *vdb,
                            vdb_comparison_func comparison_func,
                            vdb_vector *comparison_vec, unsigned int n) {
  vdb_top_n_results results;

  if (n == 0) {
    set_error_msg(vdb, "vdb: got n < 1 for vdb_top_n which is not allowed");
    results.size = 0;
    return results;
  }

  /* first calculate ALL comparisons */
  vdb_comparison_result calculated_vecs[vdb->size];

  vdb_vector *vec;
  int i;
  for (i = 0; i < vdb->size; i++) {
    vec = vdb->vector_array[i];
    if (vec) {
      vdb_comparison_result res;
      float similarity = comparison_func(comparison_vec, vec);
      res.vec_id = vec->id;
      res.similarity = similarity;
      /* may need changing below... */
      calculated_vecs[i] = res;
    }
  }

  /* qsort */
  qsort(calculated_vecs, vdb->size, sizeof(vdb_comparison_result),
        _vdb_compare);

  /* extract top n elements */
  vdb_comparison_result *top_n = malloc(sizeof(vdb_comparison_result) * n);
  if (top_n == NULL) {
    set_error_msg(vdb, "vdb: allocation failure");
    results.size = 0;
    return results;
  }

  results.vectors = top_n;
  results.size = n;

  memcpy(top_n, calculated_vecs, n * sizeof(vdb_comparison_result));

  /* return out lovely results */
  return results;
}

/* measures cosine similarity between two vectors */
float vdb_measure_cs(const vdb_vector *vec1, const vdb_vector *vec2) {
  /* size check */
  if (vec1->size != vec2->size) {
    /* no 'error' to be set via vdb as theres only one case scenario */
    return VDB_INVALID_CS;
  }
  /* reminder of cosine similarity:
   * (A . B)/(||A||*||B||) = cos(theta)
   */
  float res;
  vdb_embedding_t dp, mg1, mg2;
  unsigned int size;

  size = vec1->size;

  dp = calculate_dot_product(vec1->vec_embeddings, vec2->vec_embeddings, size);
  mg1 = calculate_magnitude(vec1->vec_embeddings, size);
  mg2 = calculate_magnitude(vec2->vec_embeddings, size);

  /* casting to float as we dont need insane precision */
  res = (float)(dp / (mg1 * mg2));

  return res;
}

/* measures euclidean distance between two vectors */
float vdb_measure_ed(const vdb_vector *vec1, const vdb_vector *vec2) {
  /* size check */
  if (vec1->size != vec2->size) {
    /* no 'error' to be set via vdb as theres only one case scenario */
    return VDB_INVALID_ED;
  }

  /* formulae reminder: d(p,q) = sqrt((p1-q1)^2 + (p2-q2)^2 + ... + (pd-qd)^2)
   */
  vdb_embedding_t s, squared_sums, rooted_sum;
  s = 0;
  squared_sums = 0;

  unsigned int size;
  size = vec1->size;

  int i;

  for (i = 0; i < size; i++) {
    s = (vec1->vec_embeddings[i] - vec2->vec_embeddings[i]);
    squared_sums += pow(s, 2);
  }

  rooted_sum = sqrt(squared_sums);

  /* cast to float as we do not need insane precision */
  return (float)rooted_sum;
}

/* get error from vdb instance */
char *vdb_get_error(vdb_instance *vdb) {
  if (vdb == NULL) {
    return ("vdb: potential allocation failure");
  }

  return vdb->err;
}

/* frees a non-associated vector */
void vdb_free_vec(vdb_vector *vec) {
  free(vec->vec_embeddings);
  free(vec);
}

/* frees the vector database */
void vdb_free(vdb_instance *vdb) {
  /* specific order required
   * 1. free vectors first & their embeddings
   * 2. free the vector array
   * 3. free the vector db in itself
   */
  vdb_vector *vec;
  int i;
  for (i = 0; i < vdb->size; i++) {
    vec = vdb->vector_array[i];
    if (vec) {
      vdb_free_vec(vec);
    }
  }

  free(vdb->vector_array);
  free(vdb);
}
