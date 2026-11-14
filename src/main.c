/* mini vector database -> a prototype */
#include "vdb/vdb.h"
#include <stdio.h>

int main() {
  vdb_instance *vdb = vdb_new(3);

  vdb_embedding_t ve1[3] = {0.1, 0.5, 0.4};
  vdb_embedding_t ve2[3] = {0.1, 0.3, 0.4};
  vdb_embedding_t ve3[3] = {0.7, 0.7, 0.9};

  printf("%f FIRST OF VE1\n", ve1[0]);

  /* issue in new_vec :( */
  vdb_vector *vec1 = vdb_new_vec(ve1, 3);
  vdb_vector *vec2 = vdb_new_vec(ve2, 3);
  vdb_vector *vec3 = vdb_new_vec(ve3, 3);
  vdb_vector *cvec = vdb_new_vec(ve3, 3);

  printf("%f FIRST OF VE1\n", vec1->vec_embeddings[0]);

  vdb_vec_id id1 = vdb_insert(vdb, vec1);
  vdb_vec_id id2 = vdb_insert(vdb, vec2);
  vdb_vec_id id3 = vdb_insert(vdb, vec3);

  vdb_top_n_results nres = vdb_top_n(vdb, vdb_measure_cs, cvec, 1);

  printf("nres size %d\n", nres.size);

  for (int i = 0; i < nres.size; i++) {
    printf("no. %d vec match: id is %d w/ match of %f\n", i + 1,
           nres.vectors[i].vec_id, nres.vectors[i].similarity);
  }

  printf("vdb error?: %s\n", vdb_get_error(vdb));

  vdb_free(vdb);
}
