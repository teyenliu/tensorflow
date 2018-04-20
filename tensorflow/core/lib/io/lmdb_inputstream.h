/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LIB_IO_LMDB_INPUTSTREAM_H_
#define TENSORFLOW_LIB_IO_LMDB_INPUTSTREAM_H_

#include <sys/stat.h>
#include "lmdb.h"

#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {

// A LMDBInputStream provides support for reading from a stream of key/value 
// using zlib (http://www.zlib.net/). Buffers the contents of the file.
//
// A given instance of an LMDBInputStream is NOT safe for concurrent use
// by multiple threads
class LMDBInputStream {
 public:
  // Create a LMDBInputStream for reading LMDB data
  LMDBInputStream(const string& mdb_filename);

  ~LMDBInputStream();

  Status OnWorkStarted();

  Status OnWorkFinished();

  Status ReadCursor(string* key, string* value, bool* produced,
                    bool* at_end);
  
  Status Reset();

 private:
  bool Seek(MDB_cursor_op op);

  string mdb_filename_;
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;

  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

  TF_DISALLOW_COPY_AND_ASSIGN(LMDBInputStream);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_LMDB_INPUTSTREAM_H_
